import datetime
import sys

# ================================
#   COULEURS ANSI
# ================================
class Color:
    RESET = "\033[0m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


# ================================
#   VARIABLES GLOBALES
# ================================
logmsg = ""
timemark = {}
saveDefault = False


# ================================
#   LOGGING AVEC COULEURS
# ================================
def log(msg, level="INFO", save=None, oneline=False):
    """
    msg: texte du log
    level: INFO / SUCCESS / WARN / ERROR / DEBUG
    save: True/False/None
    oneline: True pour écraser la ligne (progress-bar)
    """

    global logmsg, saveDefault

    # Choisir la couleur selon le level
    color = {
        "INFO": Color.BLUE,
        "SUCCESS": Color.GREEN,
        "WARN": Color.YELLOW,
        "ERROR": Color.RED,
        "DEBUG": Color.MAGENTA,
    }.get(level, Color.CYAN)

    # Timestamp
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format final
    text = f"{color}[{time}] [{level}] {msg}{Color.RESET}"

    # Sauvegarde dans logmsg si demandé
    if save is not None:
        if save:
            logmsg += text + "\n"
    elif saveDefault:
        logmsg += text + "\n"

    # Affichage console
    if oneline:
        print(text, end="\r")
    else:
        print(text)


# ================================
#   TIMER
# ================================
def marktime(marker):
    timemark[marker] = datetime.datetime.now()

def elapsed(marker):
    if marker not in timemark:
        return None
    return datetime.datetime.now() - timemark[marker]


# ================================
#   EXEMPLE
# ================================
if __name__ == "__main__":
    log("Logger initialized!", level="SUCCESS")
    marktime("start")

    log("Loading data...", level="INFO")
    log("Some details...", level="DEBUG")

    log("Warning: slow disk detected", level="WARN")
    log("Error: file missing", level="ERROR")

    log("Training running...", level="INFO", oneline=True)

    # simulate computation
    import time; time.sleep(1)

    print()  # break line
    log(f"Time elapsed: {elapsed('start')}", level="SUCCESS")
