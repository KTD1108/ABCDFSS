import os
import glob
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
    def __init__(self, datapath: str, transform, shot: int = 1, split: str = 'test'):
        self.shot = shot
        self.split = split
        self.transform = transform
        self.base_path = datapath

        # Auto-detect root directory containing class folders
        if os.path.exists(self.base_path):
            sample_classes = {'bus', 'pizza', 'spiderman', 'egg', 'coin', 'fox'}
            found = False
            for root, dirs, _ in os.walk(self.base_path):
                if any(c in dirs for c in sample_classes):
                    self.base_path = root
                    found = True
                    print(f"[*] FSS-1000 auto-detected base_path at: {self.base_path}")
                    break
            if not found and os.path.exists(os.path.join(self.base_path, 'FSS-1000')):
                self.base_path = os.path.join(self.base_path, 'FSS-1000')

        # Find available classes
        test_classes_on_disk = [c for c in FSS_TEST_CLASSES if os.path.isdir(os.path.join(self.base_path, c))]
        if test_classes_on_disk:
            self.classes = test_classes_on_disk
        else:
            # Fallback: any subdirectories that contain 1.jpg
            self.classes = sorted([
                d for d in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, d)) and os.path.exists(os.path.join(self.base_path, d, '1.jpg'))
            ])

        self.class_ids = list(range(len(self.classes)))

        # Build list of valid query images
        self.img_metadata = []
        for cat in self.classes:
            cat_dir = os.path.join(self.base_path, cat)
            for i in range(1, 11):
                img_p = os.path.join(cat_dir, f"{i}.jpg")
                if os.path.exists(img_p):
                    self.img_metadata.append(img_p)

    def __len__(self):
        return len(self.img_metadata) if self.img_metadata else len(self.classes)

    def __getitem__(self, idx):
        if self.img_metadata:
            query_img_path = self.img_metadata[idx % len(self.img_metadata)]
            class_name = os.path.basename(os.path.dirname(query_img_path))
            class_id = self.classes.index(class_name) if class_name in self.classes else 0
            query_id = int(os.path.splitext(os.path.basename(query_img_path))[0])

            # Sample support IDs different from query
            all_support_candidates = [i for i in range(1, 11) if i != query_id]
            s_ids = np.random.choice(all_support_candidates, self.shot, replace=False)
            class_dir = os.path.dirname(query_img_path)
        else:
            class_name = self.classes[idx % len(self.classes)]
            class_id = idx % len(self.classes)
            class_dir = os.path.join(self.base_path, class_name)
            chosen = np.random.choice(range(1, 11), 1 + self.shot, replace=False)
            query_id, s_ids = chosen[0], chosen[1:]
            query_img_path = os.path.join(class_dir, f"{query_id}.jpg")

        query_mask_path = os.path.join(class_dir, f"{query_id}.png")
        support_img_paths = [os.path.join(class_dir, f"{sid}.jpg") for sid in s_ids]
        support_mask_paths = [os.path.join(class_dir, f"{sid}.png") for sid in s_ids]

        # Load query
        q_img = Image.open(query_img_path).convert('RGB')
        q_mask = torch.tensor(np.array(Image.open(query_mask_path).convert('L')) >= 128).float()

        # Load supports
        s_imgs = [Image.open(p).convert('RGB') for p in support_img_paths]
        s_masks = [torch.tensor(np.array(Image.open(p).convert('L')) >= 128).float() for p in support_mask_paths]

        q_img_t = self.transform(q_img)
        q_mask_t = F.interpolate(q_mask.unsqueeze(0).unsqueeze(0), size=q_img_t.shape[-2:], mode='nearest').squeeze()

        s_imgs_t = torch.stack([self.transform(img) for img in s_imgs])
        s_masks_t = F.interpolate(torch.stack(s_masks).unsqueeze(1), size=s_imgs_t.shape[-2:], mode='nearest').squeeze(1)

        return {
            'query_img': q_img_t,
            'query_mask': q_mask_t,
            'support_set': (s_imgs_t, s_masks_t),
            'class_id': torch.tensor(class_id)
        }
