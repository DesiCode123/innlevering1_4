from flask import Flask, request, render_template, send_file
import pandas as pd
import io
import numpy as np
import cv2
import time  # Import time module
from compression_methods import (
    rle_encode, 
    huffman_encode, 
    arithmetic_encode, 
    cabac_encode,
    read_image,             
    rle_encode_image,       
    huffman_encode_image,
    arithmetic_encode_image,
    cabac_encode_image      
    )

app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('index.html')

# Route for image compression
@app.route('/compress_image', methods=['POST'])
def compress_image():
    if 'file' not in request.files:
        return 'Ingen fil valgt', 400
    file = request.files['file']
    if file.filename == '':
        return 'Ingen fil valgt', 400
    
    method = request.form.get('method')

    try:
        image = read_image(file)
    except ValueError as e:
        return str(e), 400
    
    original_size_bits = image.size * 8

    # Komprimer basert på valgt metode
    if method == 'RLE':
        start_time = time.time()
        pixels = image.flatten().tolist()
        encoded_data = rle_encode_image(pixels)
        encoding_time = time.time() - start_time
        compressed_size_bits = len(encoded_data) * (8 + 16)  # 8 bits for value, 16 bits for count
    elif method == 'Huffman':
        start_time = time.time()
        huffman_result = huffman_encode_image(image)
        encoding_time = time.time() - start_time
        compressed_size_bits = len(huffman_result['encoded_data'])
    elif method == 'Arithmetic':
        start_time = time.time()
        arithmetic_result = arithmetic_encode_image(image)
        encoding_time = time.time() - start_time
        compressed_size_bits = 64  # Assuming 64 bits for the encoded value
    elif method == 'Cabac':
        start_time = time.time()
        huffman_result = huffman_encode_image(image)
        cabac_result = cabac_encode_image(huffman_result['encoded_data'])
        encoding_time = time.time() - start_time
        compressed_size_bits = 64  # Assuming 64 bits for the encoded value
    else:
        return 'Ugyldig komprimeringsmetode', 400

    compression_ratio = (compressed_size_bits / original_size_bits) * 100

    # Send resultatene til en HTML-mal for visning
    return render_template('result.html', method=method, original_size=original_size_bits,
                           compressed_size=compressed_size_bits, compression_ratio=f"{compression_ratio:.2f}",
                           encoding_time=f"{encoding_time:.4f}")

# Route for Excel file compression
@app.route('/compress_file', methods=['POST'])
def compress_file():
    if 'file' not in request.files:
        return 'Ingen fil valgt', 400
    file = request.files['file']
    if file.filename == '':
        return 'Ingen fil valgt', 400

    # Les Excel-filen inn i en pandas DataFrame
    df = pd.read_excel(file)

    # Velge hvilken komprimeringsmetode brukeren har valgt
    method = request.form.get('method')

    # Komprimer basert på valgt metode
    if method == 'RLE':
        compressed_data = rle_encode(df)
        # Convert the compressed data to a DataFrame for saving
        compressed_df = pd.DataFrame.from_dict(compressed_data, orient='index')
    elif method == 'Huffman':
        compressed_result = huffman_encode(df)
        compressed_data = compressed_result['encoded_data']
    elif method == 'Arithmetic':
        compressed_result = arithmetic_encode(df)
        compressed_data = compressed_result['encoded_value']
    elif method == 'Cabac':
        compressed_result = cabac_encode(df)
        compressed_data = compressed_result['encoded_value']
    else:
        return 'Ugyldig komprimeringsmetode', 400

    # Prepare the compressed data for sending back to the user
    output = io.BytesIO()
    if method == 'RLE':
        # Save the DataFrame as CSV
        compressed_df.to_csv(output, index=False)
    else:
        # For other methods, write the compressed data to a text file
        output.write(str(compressed_data).encode('utf-8'))
    output.seek(0)

    # Set the appropriate filename and content type
    if method == 'RLE':
        filename = 'compressed_data.csv'
        mimetype = 'text/csv'
    else:
        filename = 'compressed_data.txt'
        mimetype = 'text/plain'

    # Send the compressed data back to the user
    return send_file(output, download_name=filename, as_attachment=True, mimetype=mimetype)

if __name__ == "__main__":
    app.run(debug=True)
