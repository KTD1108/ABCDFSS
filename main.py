from core import runner
import torch
import argparse

def parse_opts():
    r"""arguments"""
    parser = argparse.ArgumentParser(description='Adapt Before Comparison - A New Perspective on Cross-Domain Few-Shot Segmentation')

    parser.add_argument('--benchmark', type=str, choices=['fss', 'deepglobe', 'lung', 'isic', 'fss', 'suim'])
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--adapt-to', type=str, default='first-episode', choices=['first-episode', 'every-episode'])
    parser.add_argument('--postprocessing', type=str, default='off', choices=['off', 'dynamic', 'always'])
    parser.add_argument('--adapter-type', type=str, default='depthwise_separable_3x3', choices=['depthwise_separable_3x3', 'conv1x1'],
                        help='Adapter architecture: depthwise_separable_3x3 (proposed) or conv1x1 (baseline)')
    parser.add_argument('--fusion-mode', type=str, default='softmax_margin', choices=['softmax_margin', 'mean', 'learnable'],
                        help='Multi-layer fusion mode: softmax_margin (proposed), mean (baseline flat average), or learnable')
    parser.add_argument('--fusion-temp', type=float, default=1.0,
                        help='Softmax temperature scaling for layer fusion weights')
    parser.add_argument('--logpath', type=str, default='./logs')
    parser.add_argument('--verbosity', type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parse_opts()
    runner.set_args(args)

    dataloader = runner.makeDataloader()
    config = runner.makeConfig()
    config.featext.fit_every_episode = args.adapt_to == 'every-episode'
    feat_maker = runner.makeFeatureMaker(dataloader.dataset, config, device=device)
    average_meter = runner.AverageMeterWrapper(dataloader, device)

    for idx, batch in enumerate(dataloader):
        sseval = runner.SingleSampleEval(batch, feat_maker)
        sseval.post_proc_method = args.postprocessing
        sseval.forward()
        sseval.calc_metrics()
        average_meter.update(sseval)
        average_meter.write(idx)
    miou, fb_iou = average_meter.average_meter.compute_iou()
    final_msg = f"\n================ FINAL RESULTS ================\nBenchmark: {args.benchmark.upper()} | Shot: {args.nshot}\nFinal mIoU:   {miou:.2f}%\nFinal FB-IoU: {fb_iou:.2f}%\n================================================"
    print(final_msg)
    runner.Logger.info(final_msg)