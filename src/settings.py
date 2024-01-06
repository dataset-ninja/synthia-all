from typing import Dict, List, Optional, Union

from dataset_tools.templates import (
    AnnotationType,
    Category,
    CVTask,
    Domain,
    Industry,
    License,
    Research,
)

##################################
# * Before uploading to instance #
##################################
PROJECT_NAME: str = "SYNTHIA-AL"
PROJECT_NAME_FULL: str = "SYNTHIA: The SYNTHetic collection of Imagery and Annotations"
HIDE_DATASET = True  # set False when 100% sure about repo quality

##################################
# * After uploading to instance ##
##################################
LICENSE: License = License.CC_BY_NC_SA_3_0()
APPLICATIONS: List[Union[Industry, Domain, Research]] = [Industry.Automotive()]
CATEGORY: Category = Category.SelfDriving()

CV_TASKS: List[CVTask] = [CVTask.SemanticSegmentation(), CVTask.ObjectDetection()]
ANNOTATION_TYPES: List[AnnotationType] = [
    AnnotationType.SemanticSegmentation(),
    AnnotationType.ObjectDetection(),
]

RELEASE_DATE: Optional[str] = "2021-10-21"  # e.g. "YYYY-MM-DD"
if RELEASE_DATE is None:
    RELEASE_YEAR: int = None

HOMEPAGE_URL: str = "https://synthia-dataset.net/"
# e.g. "https://some.com/dataset/homepage"

PREVIEW_IMAGE_ID: int = 11995255
# This should be filled AFTER uploading images to instance, just ID of any image.

GITHUB_URL: str = "https://github.com/dataset-ninja/synthia-all"
# URL to GitHub repo on dataset ninja (e.g. "https://github.com/dataset-ninja/some-dataset")

##################################
### * Optional after uploading ###
##################################
DOWNLOAD_ORIGINAL_URL: Optional[Union[str, dict]] = {
    "SYNTHIA-AL-Train": "http://synthia-dataset.net/download/1135/",
    "SYNTHIA-AL-Test": "http://synthia-dataset.net/download/1137/",
    "README": "http://synthia-dataset.net/download/1139/",
}
# Optional link for downloading original dataset (e.g. "https://some.com/dataset/download")

CLASS2COLOR: Optional[Dict[str, List[str]]] = None
# If specific colors for classes are needed, fill this dict (e.g. {"class1": [255, 0, 0], "class2": [0, 255, 0]})

# If you have more than the one paper, put the most relatable link as the first element of the list
# Use dict key to specify name for a button

PAPER: Optional[
    Union[str, List[str], Dict[str, str]]
] = "https://openaccess.thecvf.com/content_ICCVW_2019/papers/CVRSUAD/Bengar_Temporal_Coherence_for_Active_Learning_in_Videos_ICCVW_2019_paper.pdf"

BLOGPOST: Optional[Union[str, List[str], Dict[str, str]]] = None
REPOSITORY: Optional[Union[str, List[str], Dict[str, str]]] = None

CITATION_URL: Optional[str] = None
AUTHORS: Optional[List[str]] = [
    "Javad Zolfaghari Bengar",
    "Abel Gonzalez-Garcia1 Gabriel Villalonga",
    "Bogdan Raducanu",
    "Hamed H. Aghdam1 Mikhail Mozerov",
    "Antonio M. Lopez",
    "Joost van de Weijer",
]


AUTHORS_CONTACTS: Optional[List[str]] = [
    "jzolfaghari@cvc.uab.es",
    "agonzalez@cvc.uab.es",
    "gvillalonga@cvc.uab.es",
    "bogdan@cvc.uab.es@cvc.uab.es",
    "haghdam@cvc.uab.es@cvc.uab.es",
    "mozerov@cvc.uab.es",
    "antonio@cvc.uab.es",
    "joost@cvc.uab.es",
]


ORGANIZATION_NAME: Optional[
    Union[str, List[str]]
] = "Computer Vision Center, Univ. Autonoma of Barcelona"
ORGANIZATION_URL: Optional[Union[str, List[str]]] = "https://www.cvc.uab.es/"

# Set '__PRETEXT__' or '__POSTTEXT__' as a key with string value to add custom text. e.g. SLYTAGSPLIT = {'__POSTTEXT__':'some text}
SLYTAGSPLIT: Optional[Dict[str, Union[List[str], str]]] = {
    "__PRETEXT__": "Additionally, original images and depth imagesare grouped by ***im_id***. Also every image contains information about its ***subfolder***. Explore it in supervisely labeling tool. Each bbox label contain its ***observation angle*** tag"
}
TAGS: Optional[List[str]] = None


SECTION_EXPLORE_CUSTOM_DATASETS: Optional[List[str]] = None

##################################
###### ? Checks. Do not edit #####
##################################


def check_names():
    fields_before_upload = [PROJECT_NAME]  # PROJECT_NAME_FULL
    if any([field is None for field in fields_before_upload]):
        raise ValueError("Please fill all fields in settings.py before uploading to instance.")


def get_settings():
    if RELEASE_DATE is not None:
        global RELEASE_YEAR
        RELEASE_YEAR = int(RELEASE_DATE.split("-")[0])

    settings = {
        "project_name": PROJECT_NAME,
        "project_name_full": PROJECT_NAME_FULL or PROJECT_NAME,
        "hide_dataset": HIDE_DATASET,
        "license": LICENSE,
        "applications": APPLICATIONS,
        "category": CATEGORY,
        "cv_tasks": CV_TASKS,
        "annotation_types": ANNOTATION_TYPES,
        "release_year": RELEASE_YEAR,
        "homepage_url": HOMEPAGE_URL,
        "preview_image_id": PREVIEW_IMAGE_ID,
        "github_url": GITHUB_URL,
    }

    if any([field is None for field in settings.values()]):
        raise ValueError("Please fill all fields in settings.py after uploading to instance.")

    settings["release_date"] = RELEASE_DATE
    settings["download_original_url"] = DOWNLOAD_ORIGINAL_URL
    settings["class2color"] = CLASS2COLOR
    settings["paper"] = PAPER
    settings["blog"] = BLOGPOST
    settings["repository"] = REPOSITORY
    settings["citation_url"] = CITATION_URL
    settings["authors"] = AUTHORS
    settings["authors_contacts"] = AUTHORS_CONTACTS
    settings["organization_name"] = ORGANIZATION_NAME
    settings["organization_url"] = ORGANIZATION_URL
    settings["slytagsplit"] = SLYTAGSPLIT
    settings["tags"] = TAGS

    settings["explore_datasets"] = SECTION_EXPLORE_CUSTOM_DATASETS

    return settings
