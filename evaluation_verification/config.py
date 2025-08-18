import os

RSEARCH_DIR = "/mnt/research/iPRoBeLab/sonymd"
FR_DATASET_LOCATION = os.path.join(RSEARCH_DIR, "FR-Datasets")
IRIS_DATSET_LOCATION = os.path.join(RSEARCH_DIR, "Iris-Dataset/")

FEATURE_SAVE_DIR = os.path.join(RSEARCH_DIR, "extracted_features", "face_recognition")
FEATURE_SAVE_DIR_RACE = os.path.join(RSEARCH_DIR, "extracted_features", "race_classification")
FEATURE_SAVE_DIR_IRIS = os.path.join(RSEARCH_DIR, "extracted_features", "iris_recognition")

LFW_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "lfw-dataset")
AGE_DB_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "agedb-dataset")
CFP_FP_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "cfp-dataset")
CFP_FP_DATASET_LOCATION_1024 = os.path.join(FR_DATASET_LOCATION, "cfp-dataset_1024")
CPLFW_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "cplfw-dataset")
VGGFACE2_DATASET_LOCATION = os.path.join(FR_DATASET_LOCATION, "VGGFace2")

DATASET_CHOICES = ["agedb_cr", "lfw", "cfp_fp_frontal", "cfp_fp_profile", "cplfw", "agedb", "vggface2", "iris"]
RESULTS_DIR = os.path.join(RSEARCH_DIR, "VLM-Benchmarking-Results")
MODEL_INFO_FILE = os.path.join(RESULTS_DIR, "model_param_counts.json")

AGEDB_CR_DATASET_LOCATION = os.path.join("/mnt/scratch/sonymd/FR-Datasets", "agedbcr-dataset")


# Settign the dataset paths for effeciency
import os

if os.path.exists('/dev/shm/.data/webface42m'):
    WebFace42M_DATASET_LOCATION = '/dev/shm/.data/webface42m'
elif os.path.exists('./.data/webface42m'):
    WebFace42M_DATASET_LOCATION = "./.data/webface42m"
else:
    raise ValueError("WebFace42M dataset location not found. Please set the correct path.")

if os.path.exists('/dev/shm/.data/LFW'):
    LFW_DATASET_LOCATION = '/dev/shm/.data/LFW'
elif os.path.exists('./.data/LFW'):
    LFW_DATASET_LOCATION = "./.data/LFW"
else:
    raise ValueError("LFW dataset location not found. Please set the correct path.")





# Loading Hugging Face token from a file
hf_token_file = os.path.join("~/.huggingface_token")
hf_token_file_path = os.path.expanduser(hf_token_file)
if os.path.exists(hf_token_file_path):
    with open(hf_token_file_path, "r") as f:
        line = f.readlines()[0]
        HF_TOKEN = line.strip().split()[-1]
        
    
