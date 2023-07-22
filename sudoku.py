import cv2
import pytesseract
import numpy as np

# Set the Tesseract path (update this with your Tesseract installation path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Programs\Tesseract\tesseract.exe"

puzzle = "sudoku5.png"


def find_largest_contour(contours, height, width):
    return max(
        contours,
        key=cv2.contourArea,
    )


def get_corners_from_contour(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    return np.squeeze(approx)


def order_points(corners):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def get_warped_image(image, corners):
    width, height = 450, 450
    rect = order_points(corners)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped


def divide_into_cells(image):
    rows, cols = 9, 9
    cell_height, cell_width = image.shape[0] // rows, image.shape[1] // cols
    cells = []

    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * cell_height, (r + 1) * cell_height
            x1, x2 = c * cell_width, (c + 1) * cell_width
            cells.append(image[y1:y2, x1:x2])

    return cells


def remove_cell_border(cell_image):
    # Remove the outer 10 pixels from each side of the cell image
    borderSize = 5
    cell_no_border = cell_image[borderSize:-borderSize, borderSize:-borderSize]

    return cell_no_border


preprocessed_image = cv2.imread(puzzle, cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(preprocessed_image, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(
    cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

height, width = preprocessed_image.shape[:2]
sudoku_grid_contour = find_largest_contour(contours, height, width)
sudoku_grid_corners = get_corners_from_contour(sudoku_grid_contour)
warped_image = get_warped_image(preprocessed_image, sudoku_grid_corners)
sudoku_cells = divide_into_cells(warped_image)
cells_without_border = [remove_cell_border(cell) for cell in sudoku_cells]

sudoku_array = [[0 for _ in range(9)] for _ in range(9)]

for row in range(9):
    for col in range(9):
        cell_image = cells_without_border[row * 9 + col]
        digit = pytesseract.image_to_string(
            cell_image, config="--psm 10 --oem 3 -c tessedit_char_whitelist=123456789"
        )
        cleaned_digit = digit.strip()
        sudoku_array[row][col] = int(cleaned_digit) if cleaned_digit.isdigit() else 0


def display_sudoku(sudoku):
    print("sudoku:")
    for row in sudoku:
        for cell in row:
            print(cell, end=" ")
        print()


display_sudoku(sudoku_array)

# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------

N = 9


def printing(arr):
    print("solution:")
    for i in range(N):
        for j in range(N):
            print(arr[i][j], end=" ")
        print()


def isSafe(grid, row, col, num):
    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


def solveSudoku(grid, row, col):
    if row == N - 1 and col == N:
        return True

    if col == N:
        row += 1
        col = 0

    if grid[row][col] > 0:
        return solveSudoku(grid, row, col + 1)
    for num in range(1, N + 1, 1):
        if isSafe(grid, row, col, num):
            grid[row][col] = num

            if solveSudoku(grid, row, col + 1):
                return True

        grid[row][col] = 0
    return False


if solveSudoku(sudoku_array, 0, 0):
    printing(sudoku_array)
else:
    print("no solution  exists ")
