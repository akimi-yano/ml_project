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
        label="Which part of your face do you want to detect?",
        widget=forms.Select(choices=PART_CHOICES)
    )
    
class UploadVideoAndChoosePartForm(forms.Form):
    video = forms.FileField()
    output_type = forms.CharField(
        label="Which part of your face do you want to detect?",
        widget=forms.Select(choices=PART_CHOICES)     
    )

    