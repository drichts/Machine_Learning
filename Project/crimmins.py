
def crimmins(data):
    new_image = data.copy()
    nrow = len(data)
    ncol = len(data[0])

    # Dark pixel adjustment

    # First Step
    # N-S
    for i in range(1, nrow):
        for j in range(ncol):
            if data[ i -1 ,j] >= (data[i ,j] + 2):
                new_image[i ,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol -1):
            if data[i , j +1] >= (data[i ,j] + 2):
                new_image[i ,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow):
        for j in range(1, ncol):
            if data[ i -1 , j -1] >= (data[i ,j] + 2):
                new_image[i ,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow):
        for j in range(ncol -1):
            if data[ i -1 , j +1] >= (data[i ,j] + 2):
                new_image[i ,j] += 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1, nrow -1):
        for j in range(ncol):
            if (data[ i -1 ,j] > data[i ,j]) and (data[i ,j] <= data[ i +1 ,j]):
                new_image[i ,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol -1):
            if (data[i , j +1] > data[i ,j]) and (data[i ,j] <= data[i , j -1]):
                new_image[i ,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow -1):
        for j in range(1, ncol -1):
            if (data[ i -1 , j -1] > data[i ,j]) and (data[i ,j] <= data[ i +1 , j +1]):
                new_image[i ,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow -1):
        for j in range(1, ncol -1):
            if (data[ i -1 , j +1] > data[i ,j]) and (data[i ,j] <= data[ i +1 , j -1]):
                new_image[i ,j] += 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1, nrow -1):
        for j in range(ncol):
            if (data[ i +1 ,j] > data[i ,j]) and (data[i ,j] <= data[ i -1 ,j]):
                new_image[i ,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol -1):
            if (data[i , j -1] > data[i ,j]) and (data[i ,j] <= data[i , j +1]):
                new_image[i ,j] += 1
    data = new_image
    # NW-SE
    for i in range(1, nrow -1):
        for j in range(1, ncol -1):
            if (data[ i +1 , j +1] > data[i ,j]) and (data[i ,j] <= data[ i -1 , j -1]):
                new_image[i ,j] += 1
    data = new_image
    # NE-SW
    for i in range(1, nrow -1):
        for j in range(1, ncol -1):
            if (data[ i +1 , j -1] > data[i ,j]) and (data[i ,j] <= data[ i -1 , j +1]):
                new_image[i ,j] += 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow -1):
        for j in range(ncol):
            if (data[ i +1 ,j] >= (data[i ,j ] +2)):
                new_image[i ,j] += 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1 ,ncol):
            if (data[i , j -1] >= (data[i ,j ] +2)):
                new_image[i ,j] += 1
    data = new_image
    # NW-SE
    for i in range(nrow -1):
        for j in range(ncol -1):
            if (data[ i +1 , j +1] >= (data[i ,j ] +2)):
                new_image[i ,j] += 1
    data = new_image
    # NE-SW
    for i in range(nrow -1):
        for j in range(1 ,ncol):
            if (data[ i +1 , j -1] >= (data[i ,j ] +2)):
                new_image[i ,j] += 1
    data = new_image

    # Light pixel adjustment

    # First Step
    # N-S
    for i in range(1 ,nrow):
        for j in range(ncol):
            if (data[ i -1 ,j] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(ncol -1):
            if (data[i , j +1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1 ,nrow):
        for j in range(1 ,ncol):
            if (data[ i -1 , j -1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1 ,nrow):
        for j in range(ncol -1):
            if (data[ i -1 , j +1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # Second Step
    # N-S
    for i in range(1 ,nrow -1):
        for j in range(ncol):
            if (data[ i -1 ,j] < data[i ,j]) and (data[i ,j] >= data[ i +1 ,j]):
                new_image[i ,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1, ncol -1):
            if (data[i , j +1] < data[i ,j]) and (data[i ,j] >= data[i , j -1]):
                new_image[i ,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1 ,nrow -1):
        for j in range(1 ,ncol -1):
            if (data[ i -1 , j -1] < data[i ,j]) and (data[i ,j] >= data[ i +1 , j +1]):
                new_image[i ,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1 ,nrow -1):
        for j in range(1 ,ncol -1):
            if (data[ i -1 , j +1] < data[i ,j]) and (data[i ,j] >= data[ i +1 , j -1]):
                new_image[i ,j] -= 1
    data = new_image
    # Third Step
    # N-S
    for i in range(1 ,nrow -1):
        for j in range(ncol):
            if (data[ i +1 ,j] < data[i ,j]) and (data[i ,j] >= data[ i -1 ,j]):
                new_image[i ,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1 ,ncol -1):
            if (data[i , j -1] < data[i ,j]) and (data[i ,j] >= data[i , j +1]):
                new_image[i ,j] -= 1
    data = new_image
    # NW-SE
    for i in range(1 ,nrow -1):
        for j in range(1 ,ncol -1):
            if (data[ i +1 , j +1] < data[i ,j]) and (data[i ,j] >= data[ i -1 , j -1]):
                new_image[i ,j] -= 1
    data = new_image
    # NE-SW
    for i in range(1 ,nrow -1):
        for j in range(1 ,ncol -1):
            if (data[ i +1 , j -1] < data[i ,j]) and (data[i ,j] >= data[ i -1 , j +1]):
                new_image[i ,j] -= 1
    data = new_image
    # Fourth Step
    # N-S
    for i in range(nrow -1):
        for j in range(ncol):
            if (data[ i +1 ,j] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # E-W
    for i in range(nrow):
        for j in range(1 ,ncol):
            if (data[i , j -1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # NW-SE
    for i in range(nrow -1):
        for j in range(ncol -1):
            if (data[ i +1 , j +1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    # NE-SW
    for i in range(nrow -1):
        for j in range(1 ,ncol):
            if (data[ i +1 , j -1] <= (data[i ,j ] -2)):
                new_image[i ,j] -= 1
    data = new_image
    return new_image.copy()

