
import os
from os import path
import sys
import tkinter as tk
import tkinter.ttk as ttk

from tkinter.constants import *
from PIL import ImageTk, Image
from source import about_header_instructions, about_footer_instructions, version

import DenseNetController


class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.configure('.', font="TkDefaultFont")
        self.style.map('.', background=
        [('selected', _compcolor), ('active', _ana2color)])

        top.geometry("1050x641+387+181")
        top.minsize(120, 1)
        top.maxsize(5764, 1041)
        top.resizable(0, 0)
        top.title("Link Prediction Gui")
        top.configure(background="#d9d9d9")

        self.top = top
        self.combobox = tk.StringVar()

        self.Labelframe1 = tk.LabelFrame(self.top)
        self.Labelframe1.place(x=-10, y=-20, height=674, width=342)
        self.Labelframe1.configure(relief='groove')
        self.Labelframe1.configure(foreground="#8bf44d")
        self.Labelframe1.configure(background="#7b5e73")
        self.Labelframe1.configure(cursor="fleur")
        self.Labelframe1.configure(highlightcolor="#914862")

        self.Label1 = tk.Label(self.Labelframe1)
        self.Label1.place(x=-10, y=120, height=42, width=351
                          , bordermode='ignore')
        self.Label1.configure(background="#060204")
        self.Label1.configure(compound='left')
        self.Label1.configure(cursor="fleur")
        self.Label1.configure(disabledforeground="#a3a3a3")
        self.Label1.configure(font="-family {Felix Titling} -size 20 -weight bold")
        self.Label1.configure(foreground="#ffffff")
        self.Label1.configure(text='''Link Prediction''')

        self.here = path.abspath(path.dirname(__file__))
        self.logoPath = path.join(self.here, "Braude-Logo-Horizontal-English.png")
        self.upload_logo = Image.open(self.logoPath)
        self.resized_logo = self.upload_logo.resize((300, 100), Image.AFFINE)
        self.photoimage_logo = ImageTk.PhotoImage(self.resized_logo)
        self.labeled_logo = tk.Label(self.Labelframe1,
                                     image=self.photoimage_logo, bg="black")
        self.labeled_logo.image = self.photoimage_logo
        self.labeled_logo.pack(side="left", padx=20, anchor="ne", ipadx=100)

        global _images
        _images = (

            tk.PhotoImage("img_close", data='''R0lGODlhDAAMAIQUADIyMjc3Nzk5OT09PT
                 8/P0JCQkVFRU1NTU5OTlFRUVZWVmBgYGF hYWlpaXt7e6CgoLm5ucLCwszMzNbW
                 1v//////////////////////////////////// ///////////yH5BAEKAB8ALA
                 AAAAAMAAwAAAUt4CeOZGmaA5mSyQCIwhCUSwEIxHHW+ fkxBgPiBDwshCWHQfc5
                 KkoNUtRHpYYAADs= '''),

            tk.PhotoImage("img_closeactive", data='''R0lGODlhDAAMAIQcALwuEtIzFL46
                 INY0Fdk2FsQ8IdhAI9pAIttCJNlKLtpLL9pMMMNTP cVTPdpZQOBbQd60rN+1rf
                 Czp+zLxPbMxPLX0vHY0/fY0/rm4vvx8Pvy8fzy8P//////// ///////yH5BAEK
                 AB8ALAAAAAAMAAwAAAVHYLQQZEkukWKuxEgg1EPCcilx24NcHGYWFhx P0zANBE
                 GOhhFYGSocTsax2imDOdNtiez9JszjpEg4EAaA5jlNUEASLFICEgIAOw== '''),

            tk.PhotoImage("img_closepressed", data='''R0lGODlhDAAMAIQeAJ8nD64qELE
                 rELMsEqIyG6cyG7U1HLY2HrY3HrhBKrlCK6pGM7lD LKtHM7pKNL5MNtiViNaon
                 +GqoNSyq9WzrNyyqtuzq+O0que/t+bIwubJw+vJw+vTz+zT z////////yH5BAE
                 KAB8ALAAAAAAMAAwAAAVJIMUMZEkylGKuwzgc0kPCcgl123NcHWYW Fs6Gp2mYB
                 IRgR7MIrAwVDifjWO2WwZzpxkxyfKVCpImMGAeIgQDgVLMHikmCRUpMQgA7 ''')
        )

        self.style.element_create("close", "image", "img_close",
                                  ("active", "pressed", "!disabled", "img_closepressed"),
                                  ("active", "alternate", "!disabled",
                                   "img_closeactive"), border=8, sticky='')

        self.style.layout("ClosetabNotebook", [("ClosetabNotebook.client",
                                                {"sticky": "nswe"})])
        self.style.layout("ClosetabNotebook.Tab", [
            ("ClosetabNotebook.tab",
             {"sticky": "nswe",
              "children": [
                  ("ClosetabNotebook.padding", {
                      "side": "top",
                      "sticky": "nswe",
                      "children": [
                          ("ClosetabNotebook.focus", {
                              "side": "top",
                              "sticky": "nswe",
                              "children": [
                                  ("ClosetabNotebook.label", {"side":
                                                                  "left", "sticky": ''}),
                              ]})]})]})])

        PNOTEBOOK = "ClosetabNotebook"

        self.style.configure('TNotebook.Tab', background=_bgcolor)
        self.style.configure('TNotebook.Tab', foreground=_fgcolor)
        self.style.map('TNotebook.Tab', background=
        [('selected', _compcolor), ('active', _ana2color)])
        self.NoteBook = ttk.Notebook(self.top)
        self.NoteBook.place(x=350, y=0, height=666, width=694)
        self.NoteBook.configure(takefocus="")
        self.NoteBook.configure(style=PNOTEBOOK)
        self.PreProcessing = tk.Frame(self.NoteBook)
        self.NoteBook.add(self.PreProcessing, padding=3)
        self.NoteBook.tab(0, text='''Pre Processing''', compound="left"
                          , underline='''-5''', )
        self.PreProcessing.configure(background="#d9d9d9")
        self.PreProcessing.configure(highlightbackground="#d9d9d9")
        self.PreProcessing.configure(highlightcolor="black")
        self.TrainModel = tk.Frame(self.NoteBook)
        self.NoteBook.add(self.TrainModel, padding=3)
        self.NoteBook.tab(1, text='''Train Model''', compound="left"
                          , underline='''-1''', )
        self.TrainModel.configure(background="#d9d9d9")
        self.TrainModel.configure(highlightbackground="#d9d9d9")
        self.TrainModel.configure(highlightcolor="black")
        self.LoadModel = tk.Frame(self.NoteBook)
        self.NoteBook.add(self.LoadModel, padding=3)
        self.NoteBook.tab(2, text='''Test Model''', compound="left"
                          , underline='''-1''', )
        self.LoadModel.configure(background="#d9d9d9")
        self.LoadModel.configure(highlightbackground="#d9d9d9")
        self.LoadModel.configure(highlightcolor="black")


        self.PreProcessingInputsFrame = ttk.Frame(self.PreProcessing)
        self.PreProcessingInputsFrame.place(x=10, y=50, height=165, width=655)
        self.PreProcessingInputsFrame.configure(relief='groove')
        self.PreProcessingInputsFrame.configure(borderwidth="2")
        self.PreProcessingInputsFrame.configure(relief="groove")
        self.PreProcessingInputsFrame.configure(cursor="fleur")

        self.Label2 = tk.Label(self.PreProcessingInputsFrame)
        self.Label2.place(x=10, y=10, height=21, width=13)
        self.Label2.configure(anchor='w')
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(compound='left')
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''K''')

        self.Label3 = tk.Label(self.PreProcessingInputsFrame)
        self.Label3.place(x=10, y=50, height=21, width=130)
        self.Label3.configure(anchor='w')
        self.Label3.configure(background="#d9d9d9")
        self.Label3.configure(compound='left')
        self.Label3.configure(disabledforeground="#a3a3a3")
        self.Label3.configure(foreground="#000000")
        self.Label3.configure(text='''Dimenstion Of Features''')

        self.Dimenstion_entry = tk.Entry(self.PreProcessingInputsFrame)
        self.Dimenstion_entry.place(x=160, y=50, height=20, width=134)
        self.Dimenstion_entry.configure(background="#e8fafd")
        self.Dimenstion_entry.configure(disabledforeground="#a3a3a3")
        self.Dimenstion_entry.configure(font="TkFixedFont")
        self.Dimenstion_entry.configure(foreground="#000000")
        self.Dimenstion_entry.configure(insertbackground="black")

        self.TSeparator1 = ttk.Separator(self.PreProcessingInputsFrame)
        self.TSeparator1.place(x=0, y=40, width=300)

        self.TSeparator2 = ttk.Separator(self.PreProcessingInputsFrame)
        self.TSeparator2.place(x=0, y=80, width=300)

        self.Label4 = tk.Label(self.PreProcessingInputsFrame)
        self.Label4.place(x=10, y=90, height=21, width=100)
        self.Label4.configure(anchor='w')
        self.Label4.configure(background="#d9d9d9")
        self.Label4.configure(compound='left')
        self.Label4.configure(disabledforeground="#a3a3a3")
        self.Label4.configure(foreground="#000000")
        self.Label4.configure(text='''H-hop Distance''')

        self.DistanceEntry = tk.Entry(self.PreProcessingInputsFrame)
        self.DistanceEntry.place(x=160, y=90, height=20, width=134)
        self.DistanceEntry.configure(background="#e8fafd")
        self.DistanceEntry.configure(disabledforeground="#a3a3a3")
        self.DistanceEntry.configure(font="TkFixedFont")
        self.DistanceEntry.configure(foreground="#000000")
        self.DistanceEntry.configure(insertbackground="black")

        self.TSeparator3 = ttk.Separator(self.PreProcessingInputsFrame)
        self.TSeparator3.place(x=0, y=120, width=300)

        self.Label5 = tk.Label(self.PreProcessingInputsFrame)
        self.Label5.place(x=10, y=130, height=21, width=152)
        self.Label5.configure(anchor='w')
        self.Label5.configure(background="#d9d9d9")
        self.Label5.configure(compound='left')
        self.Label5.configure(cursor="fleur")
        self.Label5.configure(disabledforeground="#a3a3a3")
        self.Label5.configure(foreground="#000000")
        self.Label5.configure(text='''Removed Edges Precentage''')

        self.TSeparator4 = ttk.Separator(self.PreProcessingInputsFrame)
        self.TSeparator4.place(x=300, y=0, height=240)
        self.TSeparator4.configure(orient="vertical")

        self.RemovedEdgePrecentage_entry = tk.Entry(self.PreProcessingInputsFrame)
        self.RemovedEdgePrecentage_entry.place(x=160, y=130, height=20
                                               , width=134)
        self.RemovedEdgePrecentage_entry.configure(background="#e8fafd")
        self.RemovedEdgePrecentage_entry.configure(disabledforeground="#a3a3a3")
        self.RemovedEdgePrecentage_entry.configure(font="TkFixedFont")
        self.RemovedEdgePrecentage_entry.configure(foreground="#000000")
        self.RemovedEdgePrecentage_entry.configure(insertbackground="black")

        self.LoadGraphBtn = tk.Button(self.PreProcessingInputsFrame)
        self.LoadGraphBtn.place(x=340, y=30, height=44, width=277)
        self.LoadGraphBtn.configure(activebackground="#ececec")
        self.LoadGraphBtn.configure(activeforeground="#000000")
        self.LoadGraphBtn.configure(background="#3f7a54")
        self.LoadGraphBtn.configure(command=DenseNetController.LoadNewGraph)
        self.LoadGraphBtn.configure(compound='left')
        self.LoadGraphBtn.configure(cursor="fleur")
        self.LoadGraphBtn.configure(disabledforeground="#a3a3a3")
        self.LoadGraphBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.LoadGraphBtn.configure(foreground="#ffffff")
        self.LoadGraphBtn.configure(highlightbackground="#21b843")
        self.LoadGraphBtn.configure(highlightcolor="#0e21cb")
        self.LoadGraphBtn.configure(pady="0")
        self.LoadGraphBtn.configure(state='active')
        self.LoadGraphBtn.configure(text='''Load New Graph''')

        self.TCombobox1 = ttk.Combobox(self.PreProcessingInputsFrame)
        self.TCombobox1.place(x=160, y=10, height=21, width=133)
        self.value_list = ['16', '32', '64', '128', '256', '512', '1024', ]
        self.TCombobox1.configure(values=self.value_list)
        self.TCombobox1.configure(justify='center')
        self.TCombobox1.configure(textvariable=self.combobox)
        self.TCombobox1.configure(validate="focusin")
        self.TCombobox1.configure(foreground="#fff0f0")
        self.TCombobox1.configure(background="#e8fafd")
        #self.TCombobox1.configure(takefocus="")
        self.TCombobox1.configure(cursor="fleur")
        self.TCombobox1['state'] = 'readonly'
        self.Label6 = tk.Label(self.PreProcessing)
        self.Label6.place(x=180, y=10, height=31, width=274)
        self.Label6.configure(background="#d9d9d9")
        self.Label6.configure(borderwidth="5")
        self.Label6.configure(compound='left')
        self.Label6.configure(disabledforeground="#a3a3a3")
        self.Label6.configure(font="-family {Tahoma} -size 20 -slant italic")
        self.Label6.configure(foreground="#000000")
        self.Label6.configure(text='''Pre Proccessing''')

        self.StartPreBtn = tk.Button(self.PreProcessing)
        self.StartPreBtn.place(x=170, y=230, height=34, width=307)
        self.StartPreBtn.configure(activebackground="#ececec")
        self.StartPreBtn.configure(activeforeground="#000000")
        self.StartPreBtn.configure(background="#3f7a54")
        self.StartPreBtn.configure(command=DenseNetController.RunPreProccessing)
        self.StartPreBtn.configure(compound='left')
        self.StartPreBtn.configure(disabledforeground="#a3a3a3")
        self.StartPreBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.StartPreBtn.configure(foreground="#ffffff")
        self.StartPreBtn.configure(highlightbackground="#d9d9d9")
        self.StartPreBtn.configure(highlightcolor="black")
        self.StartPreBtn.configure(pady="0")
        self.StartPreBtn.configure(state='active')
        self.StartPreBtn.configure(text='''Start Pre Proccessing''')

        self.TProgressbar1 = ttk.Progressbar(self.PreProcessing, orient=HORIZONTAL, length=660, mode='determinate')
        self.TProgressbar1.place(x=10, y=280, width=660, height=22)
        self.TProgressbar1['value'] = 1

        self.TFrame1 = ttk.Frame(self.PreProcessing)
        self.TFrame1.place(x=90, y=330, height=255, width=465)
        self.TFrame1.configure(relief='groove')
        self.TFrame1.configure(borderwidth="2")
        self.TFrame1.configure(relief="groove")

        self.PreProccessingList = tk.Listbox(self.TFrame1)
        self.PreProccessingList.place(x=0, y=0, height=255, width=465)
        self.PreProccessingList.configure(width=0,height=0)
        self.PreProccessingList.configure(background="#bab8be")
        self.PreProccessingList.configure(cursor="fleur")
        self.PreProccessingList.configure(disabledforeground="#a3a3a3")
        self.PreProccessingList.configure(font="-family {Tahoma} -size 10")
        self.PreProccessingList.configure(foreground="#000000")

        self.ExportPreBtn = tk.Button(self.PreProcessing)
        self.ExportPreBtn.place(x=170, y=590, height=24, width=317)
        self.ExportPreBtn.configure(activebackground="#ececec")
        self.ExportPreBtn.configure(activeforeground="#000000")
        self.ExportPreBtn.configure(background="#3f7a54")
        self.ExportPreBtn.configure(command=DenseNetController.ExportPreProccessing)
        self.ExportPreBtn.configure(compound='left')
        self.ExportPreBtn.configure(cursor="fleur")
        self.ExportPreBtn.configure(disabledforeground="#a3a3a3")
        self.ExportPreBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.ExportPreBtn.configure(foreground="#ffffff")
        self.ExportPreBtn.configure(highlightbackground="#d9d9d9")
        self.ExportPreBtn.configure(highlightcolor="black")
        self.ExportPreBtn.configure(pady="0")
        self.ExportPreBtn.configure(state='active')
        self.ExportPreBtn.configure(text='''Export Data''')

        self.Label7 = tk.Label(self.PreProcessing)
        self.Label7.place(x=220, y=305, height=25, width=204)
        self.Label7.configure(anchor='c')
        self.Label7.configure(background="#d9d9d9")
        self.Label7.configure(compound='left')
        self.Label7.configure(disabledforeground="#a3a3a3")
        self.Label7.configure(font="-family {Tahoma} -size 14")
        self.Label7.configure(foreground="#000000")
        self.Label7.configure(text='''PreProcessing''')
        self.NoteBook.bind('<Button-1>', _button_press)
        self.NoteBook.bind('<ButtonRelease-1>', _button_release)
        self.NoteBook.bind('<Motion>', _mouse_over)

        self.TrainModelInputsFrame = ttk.Frame(self.TrainModel)
        self.TrainModelInputsFrame.place(x=10, y=50, height=165, width=655)
        self.TrainModelInputsFrame.configure(relief='groove')
        self.TrainModelInputsFrame.configure(borderwidth="2")
        self.TrainModelInputsFrame.configure(relief="groove")
        self.TrainModelInputsFrame.configure(cursor="fleur")

        self.Label11 = tk.Label(self.TrainModel)
        self.Label11.place(x=180, y=10, height=31, width=274)
        self.Label11.configure(background="#d9d9d9")
        self.Label11.configure(borderwidth="5")
        self.Label11.configure(compound='left')
        self.Label11.configure(disabledforeground="#a3a3a3")
        self.Label11.configure(font="-family {Tahoma} -size 20 -slant italic")
        self.Label11.configure(foreground="#000000")
        self.Label11.configure(text='''Train Model''')

        self.Label2 = tk.Label(self.TrainModelInputsFrame)
        self.Label2.place(x=10, y=10, height=21, width=80)
        self.Label2.configure(anchor='w')
        self.Label2.configure(background="#d9d9d9")
        self.Label2.configure(compound='left')
        self.Label2.configure(disabledforeground="#a3a3a3")
        self.Label2.configure(foreground="#000000")
        self.Label2.configure(text='''Batch size''')

        self.TSeparator5 = ttk.Separator(self.TrainModelInputsFrame)
        self.TSeparator5.place(x=0, y=40, width=300)

        self.TSeparator6 = ttk.Separator(self.TrainModelInputsFrame)
        self.TSeparator6.place(x=0, y=80, width=300)

        self.batchCombobox = ttk.Combobox(self.TrainModelInputsFrame)
        self.batchCombobox.place(x=160, y=10, height=21, width=133)
        self.value_list = ['2', '8','16', '32', '64', '128', '256', '512', '1024', ]
        self.batchCombobox.configure(values=self.value_list)
        self.batchCombobox.configure(justify='center')
        self.batchCombobox.configure(textvariable=self.combobox)
        self.batchCombobox.configure(validate="all")
        self.batchCombobox.configure(foreground="#fff0f0")
        self.batchCombobox.configure(background="#e8fafd")
        #self.batchCombobox.configure(takefocus="")
        self.batchCombobox.configure(cursor="fleur")
        self.batchCombobox['state'] = 'readonly'

        self.TSeparator5 = ttk.Separator(self.TrainModelInputsFrame)
        self.TSeparator5.place(x=300, y=0, height=240)
        self.TSeparator5.configure(orient="vertical")

        self.TrainImportDataBtn = tk.Button(self.TrainModelInputsFrame)
        self.TrainImportDataBtn.place(x=340, y=30, height=44, width=277)
        self.TrainImportDataBtn.configure(activebackground="#ececec")
        self.TrainImportDataBtn.configure(activeforeground="#000000")
        self.TrainImportDataBtn.configure(background="#3f7a54")
        self.TrainImportDataBtn.configure(command=DenseNetController.TrainImportData)
        self.TrainImportDataBtn.configure(compound='left')
        self.TrainImportDataBtn.configure(cursor="fleur")
        self.TrainImportDataBtn.configure(disabledforeground="#a3a3a3")
        self.TrainImportDataBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.TrainImportDataBtn.configure(foreground="#ffffff")
        self.TrainImportDataBtn.configure(highlightbackground="#21b843")
        self.TrainImportDataBtn.configure(highlightcolor="#0e21cb")
        self.TrainImportDataBtn.configure(pady="0")
        self.TrainImportDataBtn.configure(state='active')
        self.TrainImportDataBtn.configure(text='''Import Data''')

        self.epochLabel = tk.Label(self.TrainModelInputsFrame)
        self.epochLabel.place(x=10, y=50, height=21, width=130)
        self.epochLabel.configure(anchor='w')
        self.epochLabel.configure(background="#d9d9d9")
        self.epochLabel.configure(compound='left')
        self.epochLabel.configure(disabledforeground="#a3a3a3")
        self.epochLabel.configure(foreground="#000000")
        self.epochLabel.configure(text='''epoch Number''')

        self.epoch_entry = tk.Entry(self.TrainModelInputsFrame)
        self.epoch_entry.place(x=160, y=50, height=20, width=134)
        self.epoch_entry.configure(background="#e8fafd")
        self.epoch_entry.configure(disabledforeground="#a3a3a3")
        self.epoch_entry.configure(font="TkFixedFont")
        self.epoch_entry.configure(foreground="#000000")
        self.epoch_entry.configure(insertbackground="black")

        self.TrainLabel = tk.Label(self.TrainModel)
        self.TrainLabel.place(x=300, y=310, height=21, width=204)
        self.TrainLabel.configure(anchor='w')
        self.TrainLabel.configure(background="#d9d9d9")
        self.TrainLabel.configure(compound='left')
        self.TrainLabel.configure(disabledforeground="#a3a3a3")
        self.TrainLabel.configure(font="-family {Tahoma} -size 14")
        self.TrainLabel.configure(foreground="#000000")
        self.TrainLabel.configure(text='''Results''')


        self.StartTrainBtn = tk.Button(self.TrainModel)
        self.StartTrainBtn.place(x=170, y=230, height=34, width=307)
        self.StartTrainBtn.configure(activebackground="#ececec")
        self.StartTrainBtn.configure(activeforeground="#000000")
        self.StartTrainBtn.configure(background="#3f7a54")
        self.StartTrainBtn.configure(command=DenseNetController.RunTraining)
        self.StartTrainBtn.configure(compound='left')
        self.StartTrainBtn.configure(disabledforeground="#a3a3a3")
        self.StartTrainBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.StartTrainBtn.configure(foreground="#ffffff")
        self.StartTrainBtn.configure(highlightbackground="#d9d9d9")
        self.StartTrainBtn.configure(highlightcolor="black")
        self.StartTrainBtn.configure(pady="0")
        self.StartTrainBtn.configure(state='active')
        self.StartTrainBtn.configure(text='''Start Training''')

        self.TFrame2 = ttk.Frame(self.TrainModel)
        self.TFrame2.place(x=10, y=330, height=255, width=650)
        self.TFrame2.configure(relief='groove')
        self.TFrame2.configure(borderwidth="2")
        self.TFrame2.configure(relief="groove")

        self.TrainingList = tk.Listbox(self.TFrame2)
        self.TrainingList.place(x=0, y=0, height=255, width=650)
        self.TrainingList.configure(width=0, height=0,font = 'Courier')
        self.TrainingList.configure(background="#bab8be")
        self.TrainingList.configure(cursor="fleur")
        self.TrainingList.configure(disabledforeground="#a3a3a3")
        self.TrainingList.configure(font="-family {Tahoma} -size 10")
        self.TrainingList.configure(foreground="#000000")
        self.scroolbar= tk.Scrollbar(self.TrainingList)
        self.scroolbar.pack(side=RIGHT,fill=BOTH)
        self.TrainingList.configure(yscrollcommand=self.scroolbar.set)
        self.scroolbar.config(command=self.TrainingList.yview())
        self.ExportTrainBtn = tk.Button(self.TrainModel)
        self.ExportTrainBtn.place(x=170, y=590, height=24, width=317)
        self.ExportTrainBtn.configure(activebackground="#ececec")
        self.ExportTrainBtn.configure(activeforeground="#000000")
        self.ExportTrainBtn.configure(background="#3f7a54")
        self.ExportTrainBtn.configure(command=DenseNetController.ExportTrainResults)
        self.ExportTrainBtn.configure(compound='left')
        self.ExportTrainBtn.configure(cursor="fleur")
        self.ExportTrainBtn.configure(disabledforeground="#a3a3a3")
        self.ExportTrainBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.ExportTrainBtn.configure(foreground="#ffffff")
        self.ExportTrainBtn.configure(highlightbackground="#d9d9d9")
        self.ExportTrainBtn.configure(highlightcolor="black")
        self.ExportTrainBtn.configure(pady="0")
        self.ExportTrainBtn.configure(state='active')
        self.ExportTrainBtn.configure(text='''Export Results''')

        self.LoadModelInputsFrame = ttk.Frame(self.LoadModel)
        self.LoadModelInputsFrame.place(x=10, y=50, height=165, width=655)
        self.LoadModelInputsFrame.configure(relief='groove')
        self.LoadModelInputsFrame.configure(borderwidth="2")
        self.LoadModelInputsFrame.configure(relief="groove")
        self.LoadModelInputsFrame.configure(cursor="fleur")

        self.TSeparator5 = ttk.Separator(self.LoadModelInputsFrame)
        self.TSeparator5.place(x=300, y=0, height=240)
        self.TSeparator5.configure(orient="vertical")

        self.LoadImportDataBtn = tk.Button(self.LoadModelInputsFrame)
        self.LoadImportDataBtn.place(x=340, y=30, height=44, width=277)
        self.LoadImportDataBtn.configure(activebackground="#ececec")
        self.LoadImportDataBtn.configure(activeforeground="#000000")
        self.LoadImportDataBtn.configure(background="#3f7a54")
        self.LoadImportDataBtn.configure(command=DenseNetController.LoadImportData)
        self.LoadImportDataBtn.configure(compound='left')
        self.LoadImportDataBtn.configure(cursor="fleur")
        self.LoadImportDataBtn.configure(disabledforeground="#a3a3a3")
        self.LoadImportDataBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.LoadImportDataBtn.configure(foreground="#ffffff")
        self.LoadImportDataBtn.configure(highlightbackground="#21b843")
        self.LoadImportDataBtn.configure(highlightcolor="#0e21cb")
        self.LoadImportDataBtn.configure(pady="0")
        self.LoadImportDataBtn.configure(state='active')
        self.LoadImportDataBtn.configure(text='''Import Model''')

        self.LoadLabel = tk.Label(self.LoadModel)
        self.LoadLabel.place(x=300, y=310, height=21, width=204)
        self.LoadLabel.configure(anchor='w')
        self.LoadLabel.configure(background="#d9d9d9")
        self.LoadLabel.configure(compound='left')
        self.LoadLabel.configure(disabledforeground="#a3a3a3")
        self.LoadLabel.configure(font="-family {Tahoma} -size 14")
        self.LoadLabel.configure(foreground="#000000")
        self.LoadLabel.configure(text='''Results''')

        self.Label12 = tk.Label(self.LoadModel)
        self.Label12.place(x=180, y=10, height=31, width=274)
        self.Label12.configure(background="#d9d9d9")
        self.Label12.configure(borderwidth="5")
        self.Label12.configure(compound='left')
        self.Label12.configure(disabledforeground="#a3a3a3")
        self.Label12.configure(font="-family {Tahoma} -size 20 -slant italic")
        self.Label12.configure(foreground="#000000")
        self.Label12.configure(text='''Test Model''')

        self.StartTestBtn = tk.Button(self.LoadModel)
        self.StartTestBtn.place(x=170, y=230, height=34, width=307)
        self.StartTestBtn.configure(activebackground="#ececec")
        self.StartTestBtn.configure(activeforeground="#000000")
        self.StartTestBtn.configure(background="#3f7a54")
        self.StartTestBtn.configure(command=DenseNetController.RunTest)
        self.StartTestBtn.configure(compound='left')
        self.StartTestBtn.configure(disabledforeground="#a3a3a3")
        self.StartTestBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.StartTestBtn.configure(foreground="#ffffff")
        self.StartTestBtn.configure(highlightbackground="#d9d9d9")
        self.StartTestBtn.configure(highlightcolor="black")
        self.StartTestBtn.configure(pady="0")
        self.StartTestBtn.configure(state='active')
        self.StartTestBtn.configure(text='''Start Test''')

        self.TFrame5 = ttk.Frame(self.LoadModel)
        self.TFrame5.place(x=10, y=330, height=255, width=650)
        self.TFrame5.configure(relief='groove')
        self.TFrame5.configure(borderwidth="2")
        self.TFrame5.configure(relief="groove")

        self.LoadList = tk.Listbox(self.TFrame5)
        self.LoadList.place(x=0, y=0, height=255, width=650)
        self.LoadList.configure(width=0, height=0, font='Courier')
        self.LoadList.configure(background="#bab8be")
        self.LoadList.configure(cursor="fleur")
        self.LoadList.configure(disabledforeground="#a3a3a3")
        self.LoadList.configure(font="-family {Tahoma} -size 10")
        self.LoadList.configure(foreground="#000000")
        self.scroolbar = tk.Scrollbar(self.LoadList)
        self.scroolbar.pack(side=RIGHT, fill=BOTH)
        self.LoadList.configure(yscrollcommand=self.scroolbar.set)
        self.scroolbar.config(command=self.LoadList.yview())
        self.ExportTrainBtn = tk.Button(self.LoadModel)
        self.ExportTrainBtn.place(x=170, y=590, height=24, width=317)
        self.ExportTrainBtn.configure(activebackground="#ececec")
        self.ExportTrainBtn.configure(activeforeground="#000000")
        self.ExportTrainBtn.configure(background="#3f7a54")
        self.ExportTrainBtn.configure(command=DenseNetController.ExportTestResults)
        self.ExportTrainBtn.configure(compound='left')
        self.ExportTrainBtn.configure(cursor="fleur")
        self.ExportTrainBtn.configure(disabledforeground="#a3a3a3")
        self.ExportTrainBtn.configure(font="-family {Tahoma} -size 12 -weight bold")
        self.ExportTrainBtn.configure(foreground="#ffffff")
        self.ExportTrainBtn.configure(highlightbackground="#d9d9d9")
        self.ExportTrainBtn.configure(highlightcolor="black")
        self.ExportTrainBtn.configure(pady="0")
        self.ExportTrainBtn.configure(state='active')
        self.ExportTrainBtn.configure(text='''Export Results''')


        self.menubar = tk.Menu(top, font="TkMenuFont", bg=_bgcolor, fg=_fgcolor)
        top.configure(menu=self.menubar)

        self.sub_menu = tk.Menu(top,
                                activebackground="#ececec",
                                activeborderwidth=1,
                                activeforeground="#000000",
                                background="#d9d9d9",
                                borderwidth=1,
                                disabledforeground="#a3a3a3",
                                foreground="#000000",
                                tearoff=0)
        self.menubar.add_cascade(menu=self.sub_menu,
                                 compound="left",
                                 label="Menu",
                                 state="active")
        self.sub_menu.add_command(
            command=lambda: DenseNetController.InstructionsWindow(None, "instructions", about_header_instructions,
                                                               about_footer_instructions, 300, 100),
            compound="left",
            label="Help")
        self.sub_menu.add_separator(
        )
        self.sub_menu.add_command(
            label=version,
            state="disabled"
        )
        self.sub_menu.add_separator(
        )
        self.sub_menu.add_command(
            command=DenseNetController.ExitFunc,
            compound="left",
            label="Exit")

        self.TSeparator5 = ttk.Separator(self.top)
        self.TSeparator5.place(x=340, y=0, height=650)
        self.TSeparator5.configure(orient="vertical")


# The following code is add to handle mouse events with the close icons
# in PNotebooks widgets.
def _button_press(event):
    widget = event.widget
    widget = 0



'''initialize Progress bar'''
def init_progress_bar(size):
    DenseNetController.InitProgressBar(size)


'''update progress bar'''
def update_progress_bar(size,end):
    DenseNetController.updateProgressBar(size,end)

def update_output_PreProccessing(data):
    DenseNetController.updateOutputData(data)
def _button_release(event):
    widget = event.widget
    if not widget.instate(['pressed']):
        return
    element = widget.identify(event.x, event.y)
    try:
        index = widget.index("@%d,%d" % (event.x, event.y))
    except tk.TclError:
        pass
    if "close" in element and widget._active == index:
        widget.forget(index)
        widget.event_generate("<<NotebookTabClosed>>")

    widget.state(['!pressed'])
    widget._active = None


def _mouse_over(event):
    widget = event.widget
    element = widget.identify(event.x, event.y)
    if "close" in element:
        widget.state(['alternate'])
    else:
        widget.state(['!alternate'])


def start_up():
    DenseNetController.main()


if __name__ == '__main__':
    DenseNetController.main()


def updateTrainingoutput(res):
    DenseNetController.UpdateTrainingListBox(res)

def updateLoadoutput(res):
    DenseNetController.UpdateLoadListBox(res)


def getEpoches():
    return DenseNetController.getEpoches()