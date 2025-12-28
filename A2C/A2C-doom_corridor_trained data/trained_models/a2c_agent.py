from Crypto.Util.number import long_to_bytes

# Given integer
given_integer = 11515195063862318899931685488813747395775516287289682636499965282714637259206269

# Convert the integer to bytes
hex_bytes = long_to_bytes(given_integer)

# Decode the bytes into a string
message = hex_bytes.decode('ascii')

print("Decoded Message:", message)
