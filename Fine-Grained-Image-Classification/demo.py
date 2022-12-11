import base64
import io
import os

import PySimpleGUI as sg

from PIL import Image

import torch
from torchvision import transforms

from models.modeling import ViT
from models.configs import get_b16_config
from pedia.classname import get_class_name_list


def evaluation(model, img_path):
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((600, 600), Image.BILINEAR),
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    x = Image.open(img_path)
    x = transform(x)

    # Evaluation
    with torch.no_grad():
        logits = model(x.unsqueeze(0))

    # Results
    probs = torch.nn.Softmax(dim=-1)(logits)
    top5_cls = torch.argsort(probs, dim=-1, descending=True)
    res = []
    for idx in top5_cls[0, :5]:
        res.append(idx.item() + 1)
    return res, probs


def convert_to_bytes(file_or_bytes, resize=None):
    if isinstance(file_or_bytes, str):
        img = Image.open(file_or_bytes)
    else:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), )
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()


def main():
    # Load Model
    config = get_b16_config()
    model = ViT(config)
    state_dict = torch.load('default_checkpoint.bin',
                            map_location=torch.device('cpu'))['model']
    model.load_state_dict(state_dict)
    model.eval()

    tab_name = get_class_name_list()
    # size in GUI
    image_size = (720, 720)
    pedia_size = (250, 750)
    # config of GUI
    sg.theme('GrayGrayGray')
    image_select = [[sg.Image(key='-IMAGE-', size=image_size), ],
                    [sg.In(size=(50, 1), enable_events=True, key='-FILENAME-'),
                     sg.FileBrowse(button_text="请选择图片", enable_events=True), sg.Button('确认')],
                    ]

    cate_show = [[sg.TabGroup([[sg.Tab('Tab 1', layout=[[sg.Text(size=(50, 1), key='-NAME1-')],
                                                        [sg.Image(size=pedia_size, key='-PED1-')]], key='-TAB1-'),
                                sg.Tab('Tab 2', layout=[[sg.Text(size=(50, 1), key='-NAME2-')],
                                                        [sg.Image(size=pedia_size, key='-PED2-')]], key='-TAB2-', ),
                                sg.Tab('Tab 3', layout=[[sg.Text(size=(50, 1), key='-NAME3-')],
                                                        [sg.Image(size=pedia_size, key='-PED3-')]], key='-TAB3-'),
                                sg.Tab('Tab 4', layout=[[sg.Text(size=(50, 1), key='-NAME4-')],
                                                        [sg.Image(size=pedia_size, key='-PED4-')]], key='-TAB4-'),
                                sg.Tab('Tab 5', layout=[[sg.Text(size=(50, 1), key='-NAME5-')],
                                                        [sg.Image(size=pedia_size, key='-PED5-')]], key='-TAB5-')]],
                              key='-GROUP-')]]

    layout = [[sg.Column(image_select, element_justification='c'), sg.VSeperator(),
               sg.Column(cate_show, element_justification='c')]]

    window = sg.Window('Fine-Grained Image Classification ---- demo', layout, size=(1040, 780))
    # show GUI
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        img_path = values["-FILENAME-"]
        if event in ["确认"]:
            if img_path:
                window['-IMAGE-'].update(data=convert_to_bytes(img_path, resize=image_size), size=image_size)
                top5, probs_ = evaluation(model, img_path)
                for i in range(1, 6):
                    window['-PED' + str(i) + '-'].update(
                        data=convert_to_bytes('pedia/' + str(top5[i - 1]) + '.png',
                                              resize=(pedia_size[0], pedia_size[1]-50)),
                        size=pedia_size)
                    window['-TAB' + str(i) + '-'].update(title=tab_name[top5[i - 1] - 1])
                    window['-NAME' + str(i) + '-'].update(value=tab_name[top5[i - 1] - 1] + str(probs_[-1][top5[i - 1] - 1]))

    window.close()


if __name__ == '__main__':
    main()

