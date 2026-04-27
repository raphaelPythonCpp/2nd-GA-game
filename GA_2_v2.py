import pygame
from random import randint, uniform
import torch
from math import atan, tan, cos, sqrt, ceil

class Jeu:
    def __init__(self, fenetre, police, horloge, nbJoueurs, lCouches, nbBalles, modeJeu, kNN, entrainementGA, chargementReseau):
        self.fenetre = fenetre
        self.wF, self.hF = self.fenetre.get_size()
        self.police = police
        self.horloge = horloge

        self.modeJeu = modeJeu
        self.kNN = kNN
        self.vMaxJoueurs, self.vMaxBalles = 10, 2
        self.lCouches = lCouches
        self.nbJoueurs = nbJoueurs
        self.lJoueurs = []
        for i in range(self.nbJoueurs):
            self.lJoueurs.append(Joueur(self, self.lCouches, vMax=self.vMaxJoueurs, kNN=self.kNN, iJ=i))
        if chargementReseau:
            with open("GA_2_v2.txt", 'r') as fichier:
                listeLRes = eval(fichier.read())
            _, _, meilleurNN = listeLRes
            meilleurNN = {k : torch.tensor(v, dtype=torch.float32) for k, v in meilleurNN.items()}
            self.lJoueurs[0].NN.load_state_dict(meilleurNN)
        
        self.nbBalles = nbBalles
        self.lBalles = []
        for i in range(self.nbBalles):
            if self.modeJeu == 0:
                bord = 0
            elif self.modeJeu == 1:
                bord = i%2
            self.lBalles.append(Balle(self, vMax=self.vMaxBalles, bord=bord))

        if entrainementGA:
            lNNGA = self.GA(nbJoueurs=int(input("GA : nbJoueurs = ")), nbJoueursExploration=int(input("GA : nbJoueursExploration : ")), nbBalles=self.nbBalles, nbGenerations=int(input("GA : nbGenerations = ")), nbParties=int(input("GA : nbParties = ")), nbCoupsMax=int(input("GA : nbCoupsMax = ")), tauxSurvivants=0.1)
            for i, joueur in enumerate(self.lJoueurs[:self.nbJoueurs]):
                joueur.NN.load_state_dict(lNNGA[i].state_dict())

    def reset(self, lJoueurs, lBalles):
        for joueur in lJoueurs:
            joueur.reset()
        nbBalles = len(lBalles)
        for iB, balle in enumerate(lBalles):
            if balle.bord == 0:
                yDebut = balle.lYDebut[balle.bord][0] - self.hF/nbBalles * iB
            elif balle.bord == 1:
                yDebut = balle.lYDebut[balle.bord][0] + self.hF/nbBalles * iB
            balle.reset(y=yDebut)
        return lJoueurs, lBalles

    def jouer(self, nbParties):
        for iP in range(nbParties):
            self.lJoueurs, self.lBalles = self.reset(self.lJoueurs, self.lBalles)
            lJoueurs = self.lJoueurs.copy()
            if self.jouer_partie(lJoueurs, self.lBalles, nbCoupsMax=int(input("Jeu : nbCoupsMax = "))):
                break

    def jouer_partie(self, lJoueurs, lBalles, nbCoupsMax, visuel=True, ralenti=True):
        quitterProgramme = False
        while lJoueurs and lJoueurs[0].nbCoups < nbCoupsMax:
            if ralenti:
                self.horloge.tick(100)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        lJoueurs = []
                        quitterProgramme = True
                        break
            lJoueurs = self.bouger(lJoueurs, lBalles)
            if visuel and lJoueurs:
                self.afficher(lJoueurs, lBalles)
        return quitterProgramme

    def bouger(self, lJoueurs, lBalles):
        for balle in lBalles :
            balle.bouger()
        for joueur in lJoueurs:
            joueur.jouer(lBalles)
        return [joueur for joueur in lJoueurs if joueur.enVie]

    def afficher(self, lJoueurs, lBalles):
        self.fenetre.fill((0,0,0))

        self.fenetre.blit(self.police.render(f"Score : {round(lJoueurs[0].score)} || nbCoups : {round(lJoueurs[0].nbCoups)}", True, (255,100,100,100)), (0,0))
        for joueur in lJoueurs:
            joueur.afficher()
        for balle in lBalles:
            balle.afficher()
        
        pygame.display.flip()

    def GA(self, nbJoueurs, nbJoueursExploration, nbBalles, nbGenerations, nbParties, nbCoupsMax, tauxSurvivants):
        def generer_nn_mutation(NN, epsilon):
            dico2 = {}
            for k,v in NN.state_dict().items():
                bruit = torch.randn_like(v)
                #dico2[k] = v + epsilon*bruit * (v+1)
                dico2[k] = v + epsilon*bruit
            return dico2#{k : v + v*torch.randn_like(v)*epsilon + torch.randn_like(v)*epsilon for k,v in NN.state_dict()}

        def generation(iG, lJoueurs, lBalles, nbParties, nbCoupsMax):
            lSumRes = [[iJ, 0, 0] for iJ in range(len(lJoueurs))] #[i, score, nbCoups]
            for iP in range(nbParties):
                print('\r' + ' '*80 + '\r'f"GA : Gen {iG+1} ({int(100*(iG+1)/nbGenerations)}%) || Partie {iP+1} ({int(100*(iP+1)/nbParties)}%)", end='', flush=True)
                lJoueurs, lBalles = self.reset(lJoueurs, lBalles)
                if self.jouer_partie(lJoueurs, lBalles, nbCoupsMax=nbCoupsMax, visuel=False, ralenti=False):
                    break
                for iJ, joueur in enumerate(lJoueurs):
                    lSumRes[iJ][1] += joueur.score
                    lSumRes[iJ][2] += joueur.nbCoups
            
            lSumRes.sort(key=lambda triplet : (triplet[1], triplet[2]), reverse=True)
            lRes = [(iJ, round(sommeScores/nbParties), round(sommeNbCoups/nbParties)) for iJ, sommeScores, sommeNbCoups in lSumRes]
            lJoueurs = [lJoueurs[iJ] for iJ, _, _ in lRes]
            print('\n', *[(score,nbCoups) for _, score, nbCoups in lRes])
            return lJoueurs, lRes

        torch.set_grad_enabled(False)
        #Exploration
        lJoueursExploration = []
        for i in range(nbJoueursExploration):
            lJoueursExploration.append(Joueur(self, self.lCouches, vMax=self.vMaxJoueurs, kNN=self.kNN, iJ=i))
        lBallesExploration = []
        for i in range(nbBalles):
            lBallesExploration.append(Balle(self, vMax=self.vMaxBalles, bord=0))
        lJoueursExploration, lBallesExploration = self.reset(lJoueursExploration, lBallesExploration)
        lJoueursExploration, _ = generation(-1, lJoueursExploration, lBallesExploration, nbParties, nbCoupsMax)
        #GA
        nbSurvivants = max(1, round(nbJoueurs*tauxSurvivants))
        nbMutes = ceil((nbJoueurs-nbSurvivants)/nbSurvivants)
        lJoueurs = lJoueursExploration[:nbJoueurs]
        lBalles = []
        for i in range(nbBalles):
            lBalles.append(Balle(self, vMax=self.vMaxBalles, bord=0))
        lJoueurs, lBalles = self.reset(lJoueurs, lBalles)

        epsilon, epsilonFin = 0.15, 0.001
        epsilonDecrementation = (epsilonFin-epsilon) / nbGenerations

        listeLRes = [(nbJoueurs, nbBalles, nbGenerations, nbParties, self.lCouches, self.kNN), [], []] #[caracteristiques, listeLRes, meilleurNN]
        with open("GA_2_v2.txt", 'w') as fichier:
            fichier.write(str(listeLRes))

        for iG in range(nbGenerations):
            lJoueurs, lRes = generation(iG, lJoueurs, lBalles, nbParties, nbCoupsMax)
            with open("GA_2_v2.txt", 'r') as fichier:
                listeLRes = eval(fichier.read())
            lRes = [(iJ, score, nbCoups, {'0.weight' : joueur.NN.state_dict()['0.weight'][0].detach().cpu().numpy().tolist()}) for (iJ,score,nbCoups),joueur in zip(lRes, lJoueurs)]
            listeLRes[1].append(lRes)
            listeLRes[2] = {k : v.detach().cpu().numpy().tolist() for k, v in lJoueurs[0].NN.state_dict().items()}
            with open("GA_2_v2.txt", 'w') as fichier:
                fichier.write(str(listeLRes))
            for iS, survivant in enumerate(lJoueurs[:nbSurvivants]):
                survivant.reset()
                debut = min(nbJoueurs-1, nbSurvivants + nbMutes*iS)
                for iM, mute in enumerate(lJoueurs[debut : min(nbJoueurs, debut+nbMutes)]):
                    mute.NN.load_state_dict(generer_nn_mutation(survivant.NN, epsilon))
                    mute.reset()
            epsilon -= epsilonDecrementation
        
        return [joueur.NN for joueur in lJoueurs]



        




class Joueur:
    def __init__(self, jeu, lCouches, vMax, kNN, iJ):
        self.jeu = jeu

        self.iJ = iJ

        self.enVie = None
        self.score = 0
        self.kNN = kNN

        self.r = 20
        self.x, self.y = None, None
        self.dx, self.dy = vMax, vMax
        self.couleur = (randint(128,255), randint(128,255), randint(128,255))

        lCouchesNN = []
        for i in range(1, len(lCouches)):
            lCouchesNN.append(torch.nn.Linear(lCouches[i-1], lCouches[i], bias=True))
            if i+1 < len(lCouches):
                lCouchesNN.append(torch.nn.LeakyReLU())
        self.NN = torch.nn.Sequential(*lCouchesNN)

        self.reset()

    def reset(self):
        self.enVie = True
        self.score, self.nbCoups = 0, 0
        self.x = self.jeu.wF/2 
        self.xMin, self.xMax = self.r, self.jeu.wF-self.r
        self.y = self.jeu.hF-self.r if self.jeu.modeJeu == 0 else self.jeu.hF/2 if self.jeu.modeJeu == 1 else None
        if self.jeu.modeJeu == 0:
            self.yMin, self.yMax = self.jeu.hF - self.r, self.jeu.hF - self.r
        elif self.jeu.modeJeu == 1:
            self.yMin, self.yMax = self.jeu.hF / 2, self.jeu.hF / 2

    def jouer(self, lBalles):
        lMin = sorted([((balle.x-self.x)**2+(balle.y-self.y)**2, iB) for iB, balle in enumerate(lBalles)])[:self.kNN]
        balle = lBalles[lMin[0][1]]
        if sqrt(lMin[0][0]) < self.r+balle.r:
            self.enVie = False
            return
        lInput = [self.x/self.jeu.wF]
        for k in range(self.kNN):
            balle = lBalles[lMin[k][1]]
            dx, dy = (balle.x-self.x)/self.jeu.wF, (balle.y-self.y)/self.jeu.hF
            dist = sqrt(dx**2 + dy**2)
            vx, vy = balle.dx/self.jeu.vMaxBalles, balle.dy/self.jeu.vMaxBalles
            lInput.extend([dx,dy,dist,vx,vy])
        lInput = torch.tensor(lInput, dtype=torch.float32)

        lOutput = self.NN(lInput)
        action = torch.argmax(lOutput).item()
        if action == 0: #rien
            self.x = self.x
        elif action == 1 : #gauche
            self.x = self.x-self.dx
        elif action == 2: #droite
            self.x = self.x+self.dx
        self.x = min(self.xMax, max(self.xMin, self.x))
        self.y = min(self.yMax, max(self.yMin, self.y))

        d = abs(self.x/self.jeu.wF - 0.5)
        self.score += 1 + 0.5*(1 - (4*d)**2)
        self.nbCoups += 1

    def afficher(self):
        pygame.draw.circle(self.jeu.fenetre, self.couleur, (self.x, self.y), self.r)





class Balle:
    def __init__(self, jeu, vMax, bord):
        self.jeu = jeu

        self.vMax = vMax
        self.r = 10
        self.x, self.y = None, None
        self.couleur = (randint(128,255), randint(128,255), randint(128,255))
        self.bord = bord
        self.lYDebut = [(-self.r, self.jeu.hF-self.r, self.jeu.hF+self.r), (self.jeu.hF+self.r, self.r, -self.r)] #yDebut, yFinTheorique, yFinReel
        self.reset()

    def reset(self, y=None):
        self.x = uniform(self.r, self.jeu.wF-self.r)
        if y is None:
            self.y = self.lYDebut[self.bord][0]
        else :
            self.y = y
        yFin = self.lYDebut[self.bord][1]
        xFin = uniform(self.r, self.jeu.wF-self.r)
        dx, dy = xFin-self.x, yFin-self.y
        nbDeplacements = ceil(abs(dy) / self.vMax)
        self.dy = dy / nbDeplacements
        self.dx = dx / nbDeplacements

    def bouger(self):
        self.x += self.dx
        self.y += self.dy

        if (self.bord == 0 and self.y > self.lYDebut[self.bord][2]) or (self.bord == 1 and self.y < self.lYDebut[self.bord][2]):
            self.reset()

    def afficher(self): 
        pygame.draw.circle(self.jeu.fenetre, self.couleur, (self.x, self.y), self.r)


pygame.init()

wFenetre, hFenetre = 300, 400
fenetre = pygame.display.set_mode((wFenetre, hFenetre))
pygame.display.set_caption("GA 2 Raphaël")

horloge = pygame.time.Clock()
police = pygame.font.SysFont("Arial", 10, bold=True, italic=False)

kNN = 2
env = Jeu(fenetre, police, horloge,
          nbJoueurs=10, lCouches=[1+5*kNN, 16, 3], nbBalles=4,
          modeJeu=1, 
          kNN=kNN, entrainementGA=True, chargementReseau=False)

env.jouer(nbParties=10)

pygame.quit()