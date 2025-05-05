#! /usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path
from string import Template


def split(string: str, delimiter: str) -> list[str]:
    """Split a string using delimiter. Supports escaping.

    Args:
        string (str): The string to split.
        delimiter (str): The delimiter to split the string with.

    Returns:
        list: A list of strings.
    """
    result: list[str] = []
    current = ""
    escape = False
    for char in string:
        if escape:
            current += char
            escape = False
        elif char == delimiter:
            result.append(current)
            current = ""
        elif char == "\\":
            escape = True
        else:
            current += char
    result.append(current)
    return result


def main(file_path: str | Path, substitutions: str, in_place: bool) -> None:
    # Ensure file_path is Path
    file_path_obj = Path(file_path)
    with open(file_path_obj) as f:
        template_content = f.read()

    pbtxt_template = Template(template_content)

    # Initialize sub_dict with expected types (adjust if necessary)
    sub_dict: dict[str, str | int] = {
        "max_queue_size": 0,
        "max_queue_delay_microseconds": 0,
    }
    for sub in split(substitutions, ","):
        # Assuming key is always str, value might be int/float/str after split?
        # For simplicity, treat value as str for now, convert later if needed.
        try:
            key, value = split(sub, ":")
        except ValueError:
            # Handle case where split doesn't return exactly two values
            continue
        sub_dict[key] = value  # Store value as string initially

        # Check key existence using the original template string
        assert f"${{{key}}}" in template_content or f"$ {key}" in template_content, (
            f"key '{key}' does not exist as a substitutable variable in the file {file_path_obj}."
        )

    # Perform substitution - result is a string
    substituted_pbtxt: str = pbtxt_template.safe_substitute(sub_dict)

    if in_place:
        with open(file_path_obj, "w") as f:
            f.write(substituted_pbtxt)  # Write the substituted string
    else:
        # Consider printing to stdout if not in_place?
        pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("file_path", help="path of the .pbtxt to modify")
    parser.add_argument(
        "substitutions",
        help="substitutions to perform, in the format variable_name_1:value_1,variable_name_2:value_2...",
    )
    parser.add_argument("--in_place", "-i", action="store_true", help="do the operation in-place")
    args: Namespace = parser.parse_args()
    # Ensure file_path is passed correctly, handle potential type mismatch if needed
    main(file_path=args.file_path, substitutions=args.substitutions, in_place=args.in_place)
