def remove_duplicates(data_list):
    seen = set()
    unique_data = []
    for item in data_list:
        if item not in seen:
            seen.add(item)
            unique_data.append(item)
    return unique_data