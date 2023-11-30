import os

if __name__=="__main__":
    with open("./epoch_time.csv", 'w') as f:
        for filename in os.listdir("./log"):
            rate_filename = "./log/"+filename + "/rate_result.txt"
            # lines = os.readlines(rate_filename)
            col_names = filename.split('_')
            col_names_copy = col_names.copy()
            col_names[-3] = col_names_copy[-1]
            col_names[-2] = col_names_copy[-3]
            col_names[-1] = col_names_copy[-2]
            col_name = ""
            for name in col_names:
                col_name += "_" + name
            col_name = col_name[1:]
            with open(rate_filename, 'r') as rate_file:
                lines = rate_file.readlines()
                line = lines[-5]
                results = line.split()
                result = results[-1]
                print("{}: {}".format(col_name, result))
                csv_line = col_name + ", " + result
                f.write(csv_line)
                f.write('\n')
        