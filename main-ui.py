from tkinter import *
import tkinter.ttk as ttk

from networkx.generators import directed

import main


class Application(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        master.title('Netnlyzer')
        self.experiment_options = {
            "Барабаши-Альберт" : 1,
            "Тройственное замыкание" : 2,
            "Реальная сеть из файла" : 0,
            "Пример" : 3
        }
        self.create_widgets()
        for i in range(0, 4):
            self.toggle_exp_widget_visibility(i)

        self.experiment_type_value.set("Барабаши-Альберт")


    def create_widgets(self):
        self.test_label = Label(text="Выберите тип эксперимента")
        self.test_label.grid(row=0, column=0, columnspan=2)

        self.experiment_type_value = StringVar(self.master)
        self.experiment_type_value.trace('w', self.experiment_widget_visibility)
        self.experiment_type = OptionMenu(self.master, self.experiment_type_value, *self.experiment_options.keys())
        self.experiment_type.config(width=30)
        self.experiment_type.grid(row=0, column=2, columnspan=2)
            
    def run_program(self, *args):
        # experiment_type_num=self.experiment_options[self.experiment_type_value.get()]
        # number_of_experiments=int(self.expcount_entry.get())
        # n=int(self.n_entry.get())
        # m=int(self.m_entry.get())
        # focus_indices=[int(x) for x in self.record_nodes_entry.get().split(' ')]
        # focus_period=int(self.record_period_entry.get())
        # save_data=self.write_file_value.get().__bool__()
        # value_to_analyze=self.value_analyze.get()
        # apply_log_binning=self.value_analyze_binning.get().__bool__()

        # print(experiment_type_num, number_of_experiments, n, m, focus_indices, focus_period, save_data, value_to_analyze, apply_log_binning)
        record_indices = self.record_nodes_entry.get().strip()

        self.progress_bar['value'] = 0

        main.run_external(
            experiment_type_num=self.experiment_options[self.experiment_type_value.get()],
            number_of_experiments=int(self.expcount_entry.get()),
            n=int(self.n_entry.get()),
            m=int(self.m_entry.get()),
            focus_indices=[] if not record_indices else [int(x) for x in record_indices.split(' ')],
            focus_period=int(self.record_period_entry.get()),
            save_data=self.write_file_value.get().__bool__(),
            values_to_analyze= 
                list(filter( lambda x : self.values_to_analyze[x].get() == 1
                           , self.values_to_analyze.keys() 
                           )
                    ),
            apply_log_binning=self.value_analyze_binning.get().__bool__(),
            progress_bar=self.progress_bar,
            p=float(self.p_entry.get()),
            filename=self.filename.get(),
            real_directed=self.directed_value.get().__bool__(),
        )

    def clear_experiment_widgets(self):
        if hasattr(self, "frame_experiment"):
            for item in self.frame_experiment.grid_slaves():
                item.grid_forget()

    def experiment_widget_visibility(self, *args):
        self.clear_experiment_widgets()
        exp_type_value = self.experiment_type_value.get()
        self.toggle_exp_widget_visibility(self.experiment_options[exp_type_value])

    def toggle_exp_widget_visibility(self, experiment):
        if experiment == 1:
            self.show_BA_widgets()
        elif experiment == 2:
            self.show_TC_widgets()
        elif experiment == 0:
            self.show_real_widgets()
        elif experiment == 3:
            self.show_test_widgets()

    def logbinning_widget_visibility(self, *args):
        value_type_value = self.value_analyze.get()
        if value_type_value == "beta":
            self.value_analyze_binning_label.grid(row=5, column=2)
            self.value_analyze_binning_box.grid(row=5, column=3, sticky=W)
        else:
            self.value_analyze_binning_label.grid_forget()
            self.value_analyze_binning_box.grid_forget()


    def show_simulated_widgets(self):
        self.frame_experiment = Frame(self.master, padx=5, pady=5)
        self.frame_experiment.grid(row=1, column=0, columnspan=4)
        frame = self.frame_experiment
        self.expcount_label = Label(frame, text="Кол-во экспериментов  ")
        self.expcount_label.grid(row=1, column=0, sticky=W)
        self.expcount_entry = Entry(frame)
        self.expcount_entry.insert(END, "1")
        self.expcount_entry.grid(row=1, column=1, sticky=W)

        self.n_label = Label(frame, text="n=")
        self.n_label.grid(row=2, column=0, sticky=W)
        self.n_entry = Entry(frame)
        self.n_entry.insert(END, "1000")
        self.n_entry.grid(row=2, column=1, sticky=W)
        self.m_label = Label(frame, text=" m=")
        self.m_label.grid(row=2, column=2)
        self.m_entry = Entry(frame)
        self.m_entry.insert(END, "5")
        self.m_entry.grid(row=2, column=3, sticky=W)

        self.record_nodes_label = Label(frame, text="Записывать узлы:")
        self.record_nodes_label.grid(row=3, column=0, sticky=W)
        self.record_nodes_entry = Entry(frame)
        self.record_nodes_entry.insert(END, "")
        self.record_nodes_entry.grid(row=3, column=1, sticky=W)
        self.record_period_label = Label(frame, text=" каждые ")
        self.record_period_label.grid(row=3, column=2)
        self.record_period_entry = Entry(frame)
        self.record_period_entry.insert(END, "50")
        self.record_period_entry.grid(row=3, column=3, sticky=W)
        self.record_period_label2 = Label(frame, text=" итераций")
        self.record_period_label2.grid(row=3, column=4, sticky=W)

        self.write_file_label = Label(frame, text='Запись в файл? ')
        self.write_file_label.grid(row=4, column=0, sticky=W)
        self.write_file_value = IntVar(frame, value=1)
        self.write_file_box = Checkbutton(frame, variable=self.write_file_value)
        self.write_file_box.grid(row=4, column=1, sticky=W)

        self.value_analyze_label = Label(frame, text='Знач. для анализа: ')
        self.value_analyze_label.grid(row=5, column=0, sticky=W)
        self.value_analyze = StringVar(frame)
        self.value_analyze.set("none")
        self.value_analyze.trace('w', self.logbinning_widget_visibility)
        #self.value_analyze_menu = OptionMenu(frame, self.value_analyze_menu, *["alpha", "beta", "deg-alpha", "none"])
        self.value_analyze_menu_button = Menubutton(frame, text="Выберите", 
                                     indicatoron=True, borderwidth=1, relief="raised")
        menu = Menu(self.value_analyze_menu_button, tearoff=False)
        self.value_analyze_menu_button.configure(menu=menu)
        self.value_analyze_menu_button.grid(row=5, column=1, columnspan=2, sticky=W)

        self.values_to_analyze = {}
        for choice in ("alpha", "beta", "deg-alpha"):
            self.values_to_analyze[choice] = IntVar(value=0)
            menu.add_checkbutton(label=choice, variable=self.values_to_analyze[choice], 
                                 onvalue=1, offvalue=0)
        
        self.value_analyze_binning_label = Label(frame, text='Исп. log-биннинг? ')
        self.value_analyze_binning = IntVar(frame, value=0)
        self.value_analyze_binning_box = Checkbutton(frame, variable=self.value_analyze_binning)

        self.run_button = Button(frame, text='Запуск', width=10, command=self.run_program)
        self.run_button.grid(row=6, column=2, columnspan=1, sticky=W)
        
        self.progress_bar = ttk.Progressbar(frame, mode="determinate", length=300)
        self.progress_bar.grid(row=7, column=1, columnspan=3)

        return frame

    def show_BA_widgets(self):
        self.show_simulated_widgets()

    def show_TC_widgets(self):
        frame = self.show_simulated_widgets()
        self.p_label = Label(frame, text=" p=")
        self.p_label.grid(row=1, column=2)
        self.p_entry = Entry(frame)
        self.p_entry.insert(END, "0.75")
        self.p_entry.grid(row=1, column=3, sticky=W)

    def show_real_widgets(self):
        self.frame_experiment = Frame(self.master, padx=5, pady=5)
        self.frame_experiment.grid(row=1, column=0, columnspan=4)
        frame = self.frame_experiment
        self.filename_label = Label(frame, text="Имя файла:  ")
        self.filename_label.grid(row=1, column=0, sticky=W)
        self.filename = Entry(frame, width=40)
        self.filename.insert(END, "hist_artist_edges.txt")
        self.filename.grid(row=1, column=1, columnspan=3, sticky=W)
        self.write_file_label = Label(frame, text='Запись в файл? ')
        self.write_file_label.grid(row=2, column=0, sticky=W)
        self.write_file_value = IntVar(frame, value=1)
        self.write_file_box = Checkbutton(frame, variable=self.write_file_value)
        self.write_file_box.grid(row=2, column=1, sticky=W)
        self.directed_label = Label(frame, text='Ориентированный? ')
        self.directed_label.grid(row=2, column=2, sticky=W)
        self.directed_value = IntVar(frame, value=0)
        self.directed_box = Checkbutton(frame, variable=self.directed_value)
        self.directed_box.grid(row=2, column=3, sticky=W)

        self.value_analyze_label = Label(frame, text='Знач. для анализа: ')
        self.value_analyze_label.grid(row=3, column=0, sticky=W)
        self.value_analyze = StringVar(frame)
        self.value_analyze_menu_button = Menubutton(frame, text="Выберите", 
                                     indicatoron=True, borderwidth=1, relief="raised")
        menu = Menu(self.value_analyze_menu_button, tearoff=False)
        self.value_analyze_menu_button.configure(menu=menu)
        self.value_analyze_menu_button.grid(row=3, column=1, columnspan=2, sticky=W)

        self.values_to_analyze = {}
        for choice in ("alpha", "beta", "deg-alpha"):
            self.values_to_analyze[choice] = IntVar(value=0)
            menu.add_checkbutton(label=choice, variable=self.values_to_analyze[choice], 
                                 onvalue=1, offvalue=0)

        self.value_analyze_binning_label = Label(frame, text='Исп. log-биннинг? ')
        self.value_analyze_binning = IntVar(frame, value=0)
        self.value_analyze_binning_box = Checkbutton(frame, variable=self.value_analyze_binning)

        self.run_button = Button(frame, text='Запуск', width=10, command=self.run_program)
        self.run_button.grid(row=4, column=1, columnspan=1, sticky=W)
        

    def show_test_widgets(self):
        self.frame_experiment = Frame(self.master, padx=5, pady=5)
        self.frame_experiment.grid(row=1, column=0, columnspan=4)
        frame = self.frame_experiment
        self.run_button = Button(frame, text='Запуск', width=10, command=self.run_program)
        self.run_button.grid(row=1, column=1, columnspan=1, sticky=W)
    

def run_application(**params):
    root = Tk()
    app = Application(master=root)
    #app.master.geometry("600x150")
    app.mainloop()


if __name__ == "__main__":
    run_application()
