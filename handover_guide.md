# 🚀 Handover Guide: Terminal Deployment (Git Installed)

Git est maintenant installé sur votre machine ! J'ai déjà initialisé le dépôt (`git init`) pour vous.

## Phase 1: Envoyer le code sur GitHub

Ouvrez un terminal dans ce dossier et tapez les commandes suivantes :

1.  **Préparer les fichiers** :
    ```powershell
    git add .
    git commit -m "Initial commit: Ready for Streamlit Cloud"
    ```

2.  **Lier à votre GitHub** :
    *   Créez un dépôt sur [github.com/new](https://github.com/new) nommé `autocall-pricer`.
    *   **NE cochez PAS** "Initialize with README".
    *   Copiez l'URL de votre dépôt et tapez :
    ```powershell
    git remote add origin <VOTRE_URL_GITHUB>
    git push -u origin main
    ```

## Phase 2: Déployer sur Streamlit Cloud

1.  Allez sur [share.streamlit.io](https://share.streamlit.io).
2.  Cliquez sur **"Create app"**.
3.  Sélectionnez votre dépôt GitHub.
4.  Fichier principal : `main.py`.
5.  Cliquez sur **Deploy!** 🚀

---

## 💡 Notes importantes
- **Redémarrage** : Si la commande `git` n'est pas reconnue immédiatement dans votre terminal actuel, fermez-le et rouvrez-en un nouveau (ou redémarrez votre éditeur).
- **Sécurité** : J'ai ajouté un fichier `.gitignore` pour éviter d'envoyer les fichiers inutiles (comme l'archive ZIP ou l'environnement virtuel).
- **Plan B** : Votre fichier **`project_backup.zip`** est toujours là au cas où.

---
*Vous avez maintenant le plein contrôle technique sur votre projet !*
