from datetime import date as d, datetime as dt
from tkcalendar import Calendar
from tkinter import *


def generate_calendar(prompt):

    root = Tk()
    root.geometry("400x400")
    today = str(d.today()).split("-")
    cal = Calendar(root, selectmode='day', year=int(today[0]), month=int(today[1]), day=int(today[2]))
    cal.pack(pady=20)

    def grad_date():
        date.config(text="Selected Date is: " + cal.get_date())

    def leave():
        root.quit()

    Button(root, text=prompt, command=lambda: [grad_date, leave()]).pack(pady=20)
    date = Label(root, text="")
    date.pack(pady=20)
    root.mainloop()

    date = dt.strptime(cal.get_date(), "%m/%d/%y").strftime("%Y-%m-%d")

    return date


def select_stocks():

    window = Tk()
    window.geometry("200x400")

    window.title('Stock Selection')

    yscrollbar = Scrollbar(window)
    yscrollbar.pack(side=RIGHT, fill=Y)

    label = Label(window,
                  text="Select Stocks Below :  ",
                  font=("Times New Roman", 10),
                  padx=10, pady=10)
    label.pack()
    lst = Listbox(window, selectmode="multiple",
                  yscrollcommand=yscrollbar.set)

    lst.pack(padx=10, pady=10,
             expand=YES, fill="both")

    x = ["Apple", "Microsoft", "Amazon", "Alphabet", "Facebook", "Tesla",
         "Berkshire Hathaway", "JP Morgan", "Visa", "Johnson & Johnson"]

    for each_item in range(len(x)):
        lst.insert(END, x[each_item])
        lst.itemconfig(each_item, bg="white")

    def leave():
        window.quit()

    Button(window, text="OK", command=leave).pack(pady=20)
    yscrollbar.config(command=lst.yview)
    window.mainloop()

    stocks = [lst.get(idx) for idx in lst.curselection()]

    return stocks


