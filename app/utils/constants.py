import torch

# Model input image size
IMG_SIZE = 224

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# All class names
CLASS_NAMES = [
    "Aluminium foil",
    "Bottle cap",
    "Bottle",
    "Broken glass",
    "Can",
    "Carton",
    "Cigarette",
    "Cup",
    "Lid",
    "Other litter",
    "Other plastic",
    "Paper",
    "Plastic bag - wrapper",
    "Plastic container",
    "Pop tab",
    "Straw",
    "Styrofoam piece",
    "Unlabeled litter",
    "cans",
    "cardbox",
    "food_scraps",
    "glass_like_plastic",
    "paper_material",
    "plastic",
    "AW cola",
    "Annie-s Mac Cheese",
    "Beijing Beef",
    "Chow Mein",
    "Fried Rice",
    "Hashbrown",
    "Honey Walnut Shrimp",
    "Kung Pao Chicken",
    "String Bean Chicken Breast",
    "Super Greens",
    "The Original Orange Chicken",
    "White Steamed Rice",
    "apple",
    "banana",
    "black pepper rice bowl",
    "broccoli",
    "burger",
    "carrot",
    "carrot_eggs",
    "cheese burger",
    "chicken waffle",
    "chicken_nuggets",
    "chinese_cabbage",
    "chinese_sausage",
    "crispy corn",
    "cucumber",
    "curry",
    "french fries",
    "fried chicken",
    "fried_chicken",
    "fried_dumplings",
    "fried_eggs",
    "instant_noodle",
    "juice",
    "kiwi",
    "lemon",
    "mango chicken pocket",
    "mozza burger",
    "mung_bean_sprouts",
    "nugget",
    "onion",
    "orange",
    "perkedel",
    "product",
    "rice",
    "sandwich",
    "sprite",
    "tomato",
    "tostitos cheese dip sauce",
    "triangle_hash_brown",
    "water_spinach",
]

# Group: Recyclable
RECYCLABLE_CLASSES = [
    "Aluminium foil",
    "Bottle cap",
    "Bottle",
    "Broken glass",
    "Can",
    "cans",
    "AW cola",
    "sprite",
    "juice",
    "Plastic bag - wrapper",
    "Plastic container",
    "Other plastic",
    "plastic",
    "glass_like_plastic",
    "Pop tab",
    "Styrofoam piece",
    "Paper",
    "Carton",
    "cardbox",
    "paper_material",
    "tostitos cheese dip sauce",
]

# Group: Organic 
ORGANIC_CLASSES = [
    "Carton",  
    "Cigarette",
    "Cup",
    "Lid",
    "Other litter",
    "Unlabeled litter",
    "Straw",
    "Super Greens",
    "food_scraps",
    "Annie-s Mac Cheese",
    "Beijing Beef",
    "Chow Mein",
    "Fried Rice",
    "Hashbrown",
    "Honey Walnut Shrimp",
    "Kung Pao Chicken",
    "String Bean Chicken Breast",
    "The Original Orange Chicken",
    "White Steamed Rice",
    "apple",
    "banana",
    "black pepper rice bowl",
    "broccoli",
    "burger",
    "carrot",
    "carrot_eggs",
    "cheese burger",
    "chicken waffle",
    "chicken_nuggets",
    "chinese_cabbage",
    "chinese_sausage",
    "crispy corn",
    "cucumber",
    "curry",
    "french fries",
    "fried chicken",
    "fried_chicken",
    "fried_dumplings",
    "fried_eggs",
    "instant_noodle",
    "kiwi",
    "lemon",
    "mango chicken pocket",
    "mozza burger",
    "mung_bean_sprouts",
    "nugget",
    "onion",
    "orange",
    "perkedel",
    "rice",
    "sandwich",
    "tomato",
    "triangle_hash_brown",
    "water_spinach",
]


# Class-to-group map (e.g., 'O' = Organic, 'R' = Recyclable)
CLASS_GROUP_MAP = {cls: "O" for cls in ORGANIC_CLASSES}
CLASS_GROUP_MAP.update({cls: "R" for cls in RECYCLABLE_CLASSES})

DENSENET_MODEL_PATH = "app/weights/densenet201_model.pth"
YOLO_MODEL_PATH = "app/weights/yolov8_model.pt"
BEST_MODEL_PATH = "app/weights/best_model.pt"


# Image config
WIDTH, HEIGHT = 320, 240
EXPECTED_SIZE = WIDTH * HEIGHT * 2
INFERENCE_IMAGE_SAVE_DIR = "app/data/saved_images"
TEST_IMAGE_DIR = "app/data/test_images"

DEBUG = True
SAVE_COUNT = 0
MAX_SAVE = 1

BASE_PREFIX = "/api/v1"
