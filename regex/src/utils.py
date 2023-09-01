def text_from_file(file_name):
    with open (file_name, "r", encoding='utf-8') as myfile:
        data=myfile.read()
    return data

def write_text_to_file(dest_path, text):
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.close()
    return