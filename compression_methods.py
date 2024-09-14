# compression_methods.py

from heapq import heapify, heappush, heappop
from collections import Counter
import pandas as pd
import numpy as np
import cv2

# HuffmanNode-klassen som kan brukes både for Excel-data og bilder
class HuffmanNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol  # Kan være et tegn eller en pikselverdi
        self.freq = freq
        self.left = None
        self.right = None

    # Definer sammenligningsmetode for heap
    def __lt__(self, other):
        return self.freq < other.freq

# Generell Huffman-kodingsfunksjon
def huffman_encode_generic(data_stream):
    # Beregn frekvenstabell
    freq = Counter(data_stream)

    # Bygg Huffman-treet
    heap = [HuffmanNode(symbol, frequency) for symbol, frequency in freq.items()]
    heapify(heap)

    while len(heap) > 1:
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq)
        merged.left = node1
        merged.right = node2
        heappush(heap, merged)

    huffman_tree = heap[0]

    # Generer kodebok
    codes = {}
    def generate_codes(node, current_code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = current_code
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
    generate_codes(huffman_tree)

    # Kod dataen med Huffman-koder
    encoded_data = ''.join(codes[symbol] for symbol in data_stream)

    # Returner den kodede dataen, kodeboken og treet
    return {'encoded_data': encoded_data, 'codes': codes, 'tree': huffman_tree}

# RLE-komprimering for DataFrame
def rle_encode(df):
    # RLE-komprimering for hver kolonne i DataFrame
    encoded_data = {}
    for column in df.columns:
        col_values = df[column].tolist()
        if not col_values:
            continue
        last_val = col_values[0]
        count = 1
        encoded_col = []
        for val in col_values[1:]:
            if val == last_val:
                count += 1
            else:
                encoded_col.append((last_val, count))
                last_val = val
                count = 1
        encoded_col.append((last_val, count))
        encoded_data[column] = encoded_col
    return encoded_data

# Huffman-koding for DataFrame
def huffman_encode(df):
    # Konverter DataFrame til en streng
    text = df.to_string(index=False)
    # Bruk den generelle Huffman-kodingsfunksjonen
    return huffman_encode_generic(text)

# Aritmetisk koding for DataFrame
def arithmetic_encode(df):
    # Konverter DataFrame til en streng
    text = df.to_string(index=False)

    # Frekvensanalyse
    freq = Counter(text)
    total = sum(freq.values())

    # Beregn kumulative sannsynligheter
    cumulative_prob = {}
    cumulative = 0.0
    for char, count in sorted(freq.items()):
        prob = count / total
        cumulative_prob[char] = (cumulative, cumulative + prob)
        cumulative += prob

    # Start intervallene
    low = 0.0
    high = 1.0

    # Encode teksten
    for char in text:
        low_new, high_new = cumulative_prob[char]
        range_width = high - low
        high = low + range_width * high_new
        low = low + range_width * low_new

    # Det kodede tallet er midtpunktet av [low, high]
    encoded_value = (low + high) / 2
    return {'encoded_value': encoded_value}

# CABAC-koding for DataFrame
def cabac_encode(df):
    # Konverter DataFrame til en binær streng
    text = df.to_string(index=False)
    binary_data = ''.join(format(ord(char), '08b') for char in text)

    # Start med en jevn sannsynlighet
    low = 0.0
    high = 1.0
    prob_zero = 0.5

    # Encode binær data
    for bit in binary_data:
        range_width = high - low
        mid = low + prob_zero * range_width
        if bit == '0':
            high = mid
            prob_zero += 0.01 * (1 - prob_zero)
        else:
            low = mid
            prob_zero -= 0.01 * prob_zero
        prob_zero = max(min(prob_zero, 0.99), 0.01)

    # Det kodede tallet er midtpunktet av [low, high]
    encoded_value = (low + high) / 2
    return {'encoded_value': encoded_value}

# Funksjoner for bildekomprimering

def read_image(file):
    # Les bildet ved hjelp av OpenCV
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Kunne ikke lese bildefilen.")
    return img

def rle_encode_image(pixel_values):
    # RLE-komprimering for pikselverdier
    encoded = []
    prev = pixel_values[0]
    count = 1
    for pixel in pixel_values[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1
    encoded.append((prev, count))
    return encoded

def huffman_encode_image(image):
    # Flatten bildet til en liste over pikselverdier
    pixels = image.flatten().tolist()
    # Bruk den generelle Huffman-kodingsfunksjonen
    return huffman_encode_generic(pixels)

def arithmetic_encode_image(image):
    # Flatten bildet til en liste over pikselverdier
    pixels = image.flatten().tolist()

    # Frekvensanalyse
    freq = Counter(pixels)
    total = sum(freq.values())

    # Beregn kumulative sannsynligheter
    cumulative_prob = {}
    cumulative = 0.0
    for pixel, count in sorted(freq.items()):
        prob = count / total
        cumulative_prob[pixel] = (cumulative, cumulative + prob)
        cumulative += prob

    # Start intervallene
    low = 0.0
    high = 1.0

    # Encode pikselverdiene
    for pixel in pixels:
        low_new, high_new = cumulative_prob[pixel]
        range_width = high - low
        high = low + range_width * high_new
        low = low + range_width * low_new

    # Det kodede tallet er midtpunktet av [low, high]
    encoded_value = (low + high) / 2
    return {'encoded_value': encoded_value}

def cabac_encode_image(encoded_data):
    # Forventet at encoded_data er en binær streng fra Huffman-koding
    binary_data = encoded_data

    # Start med en jevn sannsynlighet
    low = 0.0
    high = 1.0
    prob_zero = 0.5

    # Encode binær data
    for bit in binary_data:
        range_width = high - low
        mid = low + prob_zero * range_width
        if bit == '0':
            high = mid
            prob_zero += 0.01 * (1 - prob_zero)
        else:
            low = mid
            prob_zero -= 0.01 * prob_zero
        prob_zero = max(min(prob_zero, 0.99), 0.01)

    # Det kodede tallet er midtpunktet av [low, high]
    encoded_value = (low + high) / 2
    return {'encoded_value': encoded_value}
