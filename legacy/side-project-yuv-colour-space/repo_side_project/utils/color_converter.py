def lab_to_rgb(L, a, b):
    # Reference white D65
    REF_X =  95.047
    REF_Y = 100.000
    REF_Z = 108.883

    # Convert LAB to XYZ
    Y = (L + 16) / 116
    X = a / 500 + Y
    Z = Y - b / 200

    def pivot(t):
        return t ** 3 if t > 0.008856 else (t - 16/116) / 7.787

    X = REF_X * pivot(X)
    Y = REF_Y * pivot(Y)
    Z = REF_Z * pivot(Z)

    # Convert XYZ to RGB
    X /= 100
    Y /= 100
    Z /= 100

    R = X *  3.2406 + Y * -1.5372 + Z * -0.4986
    G = X * -0.9689 + Y *  1.8758 + Z *  0.0415
    B = X *  0.0557 + Y * -0.2040 + Z *  1.0570

    def gamma_correct(c):
        c = max(0, min(1, c))  # clamp
        return 1.055 * (c ** (1 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c

    R = round(gamma_correct(R) * 255)
    G = round(gamma_correct(G) * 255)
    B = round(gamma_correct(B) * 255)

    return (R, G, B)