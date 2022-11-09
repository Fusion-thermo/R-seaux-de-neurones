from tkinter import * 
import random

width=600
height=width
taille_carte=10
unite=width//(taille_carte+2)
cote_carre=unite-1
x0=unite
y0=height-unite
taux_suppr_bordures=0
couleurs=["red","blue","green","brown"]


fenetre = Tk()

Canevas = Canvas(fenetre, width=width, height=height)
Canevas.pack()

Bouton1 = Button(fenetre, text = 'Quitter', command = fenetre.destroy)
Bouton1.pack()

Canevas.create_rectangle(50,50,100,100,fill=None)


fenetre.mainloop()