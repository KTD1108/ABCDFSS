import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

# Standard 240 test classes for FSS-1000 benchmark
FSS_TEST_CLASSES = [
    "bus", "hotel_slipper", "burj_al", "reflex_camera", "abe's_flyingfish", "oiltank_car", "doormat",
    "fish_eagle", "barber_shaver", "motorbike", "feather_clothes", "wandering_albatross", "rice_cooker",
    "delta_wing", "fish", "nintendo_switch", "bustard", "diver", "minicooper", "cathedrale_paris",
    "big_ben", "combination_lock", "villa_savoye", "american_alligator", "gym_ball", "andean_condor",
    "leggings", "pyramid_cube", "jet_aircraft", "meatloaf", "reel", "swan", "osprey", "crt_screen",
    "microscope", "rubber_eraser", "arrow", "monkey", "mitten", "spiderman", "parthenon", "bat",
    "chess_king", "sulphur_butterfly", "quail_egg", "oriole", "iron_man", "wooden_boat", "anise",
    "steering_wheel", "groenendael", "dwarf_beans", "pteropus", "chalk_brush", "bloodhound", "moon",
    "english_foxhound", "boxing_gloves", "peregine_falcon", "pyraminx", "cicada", "screw", "shower_curtain",
    "tredmill", "bulb", "bell_pepper", "lemur_catta", "doughnut", "twin_tower", "astronaut",
    "nintendo_3ds", "fennel_bulb", "indri", "captain_america_shield", "kunai", "broom", "iphone",
    "earphone1", "flying_squirrel", "onion", "vinyl", "sydney_opera_house", "oyster", "harmonica",
    "egg", "breast_pump", "guitar", "potato_chips", "tunnel", "cuckoo", "rubick_cube", "plastic_bag",
    "phonograph", "net_surface_shoes", "goldfinch", "ipad", "mite_predator", "coffee_mug",
    "golden_plover", "f1_racing", "lapwing", "nintendo_gba", "pizza", "rally_car", "drilling_platform",
    "cd", "fly", "magpie_bird", "leaf_fan", "little_blue_heron", "carriage", "moist_proof_pad",
    "flying_snakes", "dart_target", "warehouse_tray", "nintendo_wiiu", "chiffon_cake", "bath_ball",
    "manatee", "cloud", "marimba", "eagle", "ruler", "soymilk_machine", "sled", "seagull",
    "glider_flyingfish", "doublebus", "transport_helicopter", "window_screen", "truss_bridge", "wasp",
    "snowman", "poached_egg", "strawberry", "spinach", "earphone2", "downy_pitch", "taj_mahal",
    "rocking_chair", "cablestayed_bridge", "sealion", "banana_boat", "pheasant", "stone_lion",
    "electronic_stove", "fox", "iguana", "rugby_ball", "hang_glider", "water_buffalo", "lotus",
    "paper_plane", "missile", "flamingo", "american_chamelon", "kart", "chinese_knot",
    "cabbage_butterfly", "key", "church", "tiltrotor", "helicopter", "french_fries", "water_heater",
    "snow_leopard", "goblet", "fan", "snowplow", "leafhopper", "pspgo", "black_bear", "quail",
    "condor", "chandelier", "hair_razor", "white_wolf", "toaster", "pidan", "pyramid", "chicken_leg",
    "letter_opener", "apple_icon", "porcupine", "chicken", "stingray", "warplane", "windmill",
    "bamboo_slip", "wig", "flying_geckos", "stonechat", "haddock", "australian_terrier", "hover_board",
    "siamang", "canton_tower", "santa_sledge", "arch_bridge", "curlew", "sushi", "beet_root",
    "accordion", "leaf_egg", "stealth_aircraft", "stork", "bucket", "hawk", "chess_queen", "ocarina",
    "knife", "whippet", "cantilever_bridge", "may_bug", "wagtail", "leather_shoes", "wheelchair",
    "shumai", "speedboat", "vacuum_cup", "chess_knight", "pumpkin_pie", "wooden_spoon",
    "bamboo_dragonfly", "ganeva_chair", "soap", "clearwing_flyingfish", "pencil_sharpener1", "cricket",
    "photocopier", "nintendo_sp", "samarra_mosque", "clam", "charge_battery", "flying_frog",
    "ferrari911", "polo_shirt", "echidna", "coin", "tower_pisa"
]

class FSS1000Dataset(Dataset):
    """Clean FSS-1000 Dataset loader with robust nested directory auto-detection."""
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test', manifest_path: str = None):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath
        self.manifest_path = manifest_path

        # Auto-detect root directory containing class folders
        if os.path.exists(self.base_path):
            test_cls_set = set(FSS_TEST_CLASSES)
            # Direct check: does self.base_path directly contain FSS classes?
            is_direct_root = False
            try:
                base_entries = set(os.listdir(self.base_path))
                if len(base_entries.intersection(test_cls_set)) >= 5:
                    is_direct_root = True
            except Exception:
                pass

            if not is_direct_root:
                found = False
                for root, dirs, files in os.walk(self.base_path):
                    if len(set(dirs).intersection(test_cls_set)) >= 5:
                        self.base_path = root
                        found = True
                        print(f"[*] FSS-1000 auto-detected base_path at: {self.base_path}")
                        break
                if not found and os.path.exists(os.path.join(self.base_path, 'FSS-1000')):
                    self.base_path = os.path.join(self.base_path, 'FSS-1000')

        # Collect class mapping: {class_name: class_dir_path}
        self.class_dirs = {}
        for c in FSS_TEST_CLASSES:
            cp = os.path.join(self.base_path, c)
            if os.path.isdir(cp):
                self.class_dirs[c] = cp

        if not self.class_dirs and os.path.exists(self.base_path):
            # Fallback: any subdirectories containing .jpg files
            for d in os.listdir(self.base_path):
                dp = os.path.join(self.base_path, d)
                if os.path.isdir(dp):
                    try:
                        if any(f.endswith('.jpg') for f in os.listdir(dp)):
                            self.class_dirs[d] = dp
                    except Exception:
                        pass

        self.classes = sorted(list(self.class_dirs.keys()))
        self.class_ids = list(range(len(self.classes)))
        self.episodes = None

        if self.manifest_path and os.path.exists(self.manifest_path):
            import json
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                self.episodes = manifest_data.get('episodes', manifest_data)
                print(f"[*] FSS1000Dataset loaded {len(self.episodes)} fixed episodes from manifest: {self.manifest_path}")
            self.img_metadata = []
        else:
            # Build list of valid query images: (img_path, mask_path, class_name)
            self.img_metadata = []
            for cat, cat_dir in self.class_dirs.items():
                for i in range(1, 11):
                    img_p = os.path.join(cat_dir, f"{i}.jpg")
                    mask_p = os.path.join(cat_dir, f"{i}.png")
                    if os.path.exists(img_p) and os.path.exists(mask_p):
                        self.img_metadata.append((img_p, mask_p, cat))

            print(f"[*] FSS-1000 loaded successfully: {len(self.classes)} classes, {len(self.img_metadata)} images.")

    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        return len(self.img_metadata) if self.img_metadata else len(self.classes)

    def __getitem__(self, idx):
        if self.episodes is not None:
            ep = self.episodes[idx]
            q_img_rel = ep['query_img'].replace('\\', '/')
            q_mask_rel = ep['query_mask'].replace('\\', '/')
            q_img_path = q_img_rel if os.path.isabs(q_img_rel) else os.path.join(self.base_path, q_img_rel)
            q_mask_path = q_mask_rel if os.path.isabs(q_mask_rel) else os.path.join(self.base_path, q_mask_rel)
            support_img_paths = [p.replace('\\', '/') if os.path.isabs(p.replace('\\', '/')) else os.path.join(self.base_path, p.replace('\\', '/')) for p in ep['support_imgs']]
            support_mask_paths = [p.replace('\\', '/') if os.path.isabs(p.replace('\\', '/')) else os.path.join(self.base_path, p.replace('\\', '/')) for p in ep['support_masks']]
            class_id = ep.get('class_id', 0)
        elif self.img_metadata:
            q_img_path, q_mask_path, class_name = self.img_metadata[idx % len(self.img_metadata)]
            class_id = self.classes.index(class_name) if class_name in self.classes else 0
            query_id = int(os.path.splitext(os.path.basename(q_img_path))[0])
            cat_dir = self.class_dirs[class_name]

            # Sample support IDs different from query
            all_support_candidates = [i for i in range(1, 11) if i != query_id]
            s_ids = np.random.choice(all_support_candidates, self.shot, replace=False)
            support_img_paths = [os.path.join(cat_dir, f"{sid}.jpg") for sid in s_ids]
            support_mask_paths = [os.path.join(cat_dir, f"{sid}.png") for sid in s_ids]
        else:
            class_name = self.classes[idx % len(self.classes)]
            class_id = idx % len(self.classes)
            cat_dir = self.class_dirs[class_name]
            chosen = np.random.choice(range(1, 11), 1 + self.shot, replace=False)
            query_id, s_ids = chosen[0], chosen[1:]
            q_img_path = os.path.join(cat_dir, f"{query_id}.jpg")
            q_mask_path = os.path.join(cat_dir, f"{query_id}.png")
            support_img_paths = [os.path.join(cat_dir, f"{sid}.jpg") for sid in s_ids]
            support_mask_paths = [os.path.join(cat_dir, f"{sid}.png") for sid in s_ids]

        # Load query
        q_img = Image.open(q_img_path).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(q_mask_path).convert('L')) >= 128).float()

        # Load supports
        s_imgs = [Image.open(p).convert('RGB') for p in support_img_paths]
        s_masks = [torch.tensor(np.array(Image.open(p).convert('L')) >= 128).float() for p in support_mask_paths]

        q_img_t = self.transform(q_img)
        q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

        s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
        s_masks_t = F.interpolate(torch.stack(s_masks).unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

        res = {
            'query_img': q_img_t,
            'query_mask': q_mask_t,
            'support_set': (s_imgs_t, s_masks_t),
            'class_id': torch.tensor(class_id)
        }
        if self.episodes is not None:
            ep = self.episodes[idx]
            res['episode_id'] = ep.get('episode_id', idx)
            res['query_img_name'] = ep.get('query_img', '')
            res['support_img_names'] = ep.get('support_imgs', [])
            res['category'] = ep.get('category', str(class_id))
        return res
