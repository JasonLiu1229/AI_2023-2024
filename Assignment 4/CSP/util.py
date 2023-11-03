from tqdm import tqdm

progressBars = dict()


def monitor(f):
    """ Decorator to time functions and count the amount of calls. """
    import json

    def create_id(file_name: str = "log.json"):
        with open(file_name, "r") as file:
            try:
                log = json.load(file)
            except:
                log = dict()
            new_run_id = len(log)
            file.close()
        with open(file_name, "w") as file:
            log[f"run_{new_run_id}"] = 0
            json.dump(log, file)
            file.close()

    def get_dictionary(file_name: str = "log.json"):
        with open(file_name, "r") as file:
            try:
                log = json.load(file)
            except:
                log = dict()
            file.close()
            return log

    def wrapper(*args, **kwargs):
        if f not in progressBars:
            progressBars[f] = tqdm(desc=f.__name__, unit=" calls")
            create_id()
        progress = progressBars[f]
        progress.update(1)
        dictionary = get_dictionary()
        run_id = len(dictionary) - 1
        dictionary[f"run_{run_id}"] += 1
        with open("log.json", "w") as file:
            json.dump(dictionary, file)
            file.close()
        return f(*args, **kwargs)

    return wrapper
