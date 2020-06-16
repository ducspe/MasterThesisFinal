class Logger:
    def __init__(self, file_name):
        self.result_file = file_name

        with open(self.result_file, "w") as f:
            f.write("Instantiated the log file\n")

    def store(self, string_info):
        with open(self.result_file, "a") as f:
            f.write(string_info)
            f.write("\n")

