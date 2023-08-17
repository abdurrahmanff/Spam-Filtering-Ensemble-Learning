from xml.dom import minidom
import codecs
import os
import subprocess

TMP_EXT = ".tmp.1"
CHAR_TO_REMOVE = "&"


def remove_special_chars(filename):
    script_path = os.path.join(os.getcwd(), "remove_chars.ps1")
    filename = os.getcwd() + filename
    with open(script_path, "w") as script_file:
        script_file.write(
            'Get-Content {} | %{{$_ -replace "{}", ""}} | Out-File -encoding ASCII {}'.format(
                filename, CHAR_TO_REMOVE, filename + TMP_EXT
            )
        )

    command = ["powershell", "-File", script_path]
    subprocess.call(command)
    os.remove(script_path)


def add_root_element(filename):
    filename = os.getcwd() + filename
    with open(filename, "r") as file:
        content = file.read()

    added_content = "<ROOT>\n" + content + "</ROOT>\n"

    with open(filename, "w") as file:
        file.write(added_content)


def main():
    files = {
        "train_ham": "\\Datasets\\GenSpam\\train_GEN.ems",
        "train_spam": "\\Datasets\\GenSpam\\train_SPAM.ems",
        "test_ham": "\\Datasets\\GenSpam\\test_GEN.ems",
        "test_spam": "\\Datasets\\GenSpam\\test_SPAM.ems",
    }

    for filename in files.values():
        remove_special_chars(filename)
        add_root_element(filename + TMP_EXT)

    for key in files.keys():
        filename = os.getcwd() + files[key] + TMP_EXT
        with open(filename, "r") as file:
            print("parsing {}: {}".format("asdasd", filename))

            doc = minidom.parse(file)

            message_list = doc.getElementsByTagName("MESSAGE")

            print(len(message_list))

            file.close()


if __name__ == "__main__":
    main()
