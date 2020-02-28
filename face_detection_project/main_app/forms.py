from django import forms

class UploadFileForm(forms.Form):
    image = forms.ImageField()

class UploadVideoForm(forms.Form):
    video =  forms.FileField()

PART_CHOICES =[ 
("mouth","mouth"),
("right_eyebrow","right_eyebrow"),
("left_eyebrow","left_eyebrow"),
("right_eye","right_eye"),
("left_eye","left_eye"),
("nose","nose"),
("jaw","jaw")
]
class UploadFileAndChoosePartForm(forms.Form):
    image = forms.ImageField()
    output_type = forms.CharField(
        label="Select a part of face",
        widget=forms.Select(choices=PART_CHOICES)
    )
    
class UploadVideoAndChoosePartForm(forms.Form):
    video = forms.FileField()
    output_type = forms.CharField(
        label="Select a part of face",
        widget=forms.Select(choices=PART_CHOICES)     
    )

IMAGE_CHOICES = [
    ("wink_detection_app/heart.png", "heart"),
    ("wink_detection_app/star.png", "star"),
    ("wink_detection_app/kinoko.png", "mushroom")
]
SLEEP_CHOICES = [
    ("sleepiness_detection_app/pikachu.png", "sleeping_pikachu"),
    ("sleepiness_detection_app/sleep.png", "sleeping"),
    ("sleepiness_detection_app/purin.png","singing_jigglypuff")
    
]
class UploadVideoAndChooseImages(forms.Form):
    video = forms.FileField()
    right_wink_image = forms.CharField(
        label="Select an image for right wink",
        widget=forms.Select(choices=IMAGE_CHOICES)     
    )
    left_wink_image = forms.CharField(
        label="Select an image for left wink",
        widget=forms.Select(choices=IMAGE_CHOICES)     
    )
    blink_image = forms.CharField(
        label="Select an image for blink",
        widget=forms.Select(choices=IMAGE_CHOICES)     
    )
    
class UploadVideoAndChooseSleeping_Image(forms.Form):
    video = forms.FileField()
    sleeping_image = forms.CharField(
        label="Select an image",
        widget=forms.Select(choices=SLEEP_CHOICES)     
    )
    
GLASSES_CHOICES = [
    ("realtime/glasses.png", "shades"),
    ("realtime/mustash.png", "moustache_glasses"),
    ("realtime/fancy_black.png", "fancy_glasses"),
    ("realtime/red_real.png", "real_glasses"),
    ("realtime/pixel.png", "pixel_glasses")
]
class UploadVideoAndChooseGlasses_Image(forms.Form):
    video = forms.FileField()
    glasses_image = forms.CharField(
        label="Select glasses",
        widget=forms.Select(choices=GLASSES_CHOICES)     
    )
    
SWAP_CHOICES = [
    ("face_swap_app/heart_emoji.png", "heart_face_emoji"),
    ("face_swap_app/crying_emoji.png", "crying_face_emoji"),
    ("face_swap_app/happy_emoji.png", "happy_face_emoji"),
    ("face_swap_app/kiss_emoji.png", "kiss_face_emoji"),
    ("face_swap_app/shades_emoji.png", "shades_face_emoji"),
    ("face_swap_app/smily_emoji.png", "smily_face_emoji"),
    ("face_swap_app/surprized_emoji.png", "surprized_face_emoji"),
    ("face_swap_app/wink_emoji.png", "wink_face_emoji")
]
class UploadImageAndChooseSwap_Image(forms.Form):
    image = forms.ImageField()
    swap_image = forms.CharField(
        label="Select a face",
        widget=forms.Select(choices=SWAP_CHOICES)     
    )

class UploadImageAndChooseGlasses_Image(forms.Form):
    image = forms.ImageField()
    glasses_image = forms.CharField(
        label="Select glasses",
        widget=forms.Select(choices=GLASSES_CHOICES)     
    )