from unstructured.partition.pdf import partition_pdf

def parse_resume_pdf(filename, output_filename="out.txt"):
    # Parse PDF document and extract elements
    elements = partition_pdf(filename)

    # Convert elements to text and join with newlines
    content = "\n".join([str(element) for element in elements])

    # Append content to output file
    with open(output_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(content)

    return (content, elements)