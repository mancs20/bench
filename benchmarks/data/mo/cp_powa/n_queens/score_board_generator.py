import os

import random


def generate_score_board(queens):
    score_board = [[0 for _ in range(queens)] for _ in range(queens)]
    for i in range(queens):
        for j in range(queens):
            score_board[i][j] = random.randint(1, 100)

    return score_board


def generate_score_board_m_obj(queens, objs):
    list_of_score_boards = []
    for i in range(objs):
        score_board = generate_score_board(queens)
        list_of_score_boards.append(score_board)
    return list_of_score_boards


def write_score_board_list(score_board_list, directory, instance_number):
    name = generate_name(len(score_board_list), len(score_board_list[0]), instance_number)
    with open(os.path.join(directory, name), "w") as file:
        file.write(f"{len(score_board_list)}\n{len(score_board_list[0])}\n")
        for i, score_board in enumerate(score_board_list):
            file.write(f"\n")
            for row in score_board:
                file.write(" ".join(str(cell) for cell in row) + "\n")

    print(f"File created: {name}")


def generate_name(objs, queens, instance_number):
    return f"n_queens_p-{objs}_q-{queens}_ins-{instance_number}.dat"


if __name__ == "__main__":
    n_queens_list = [8, 10, 12, 14]
    number_objs_list = [2, 3, 4, 5]
    number_instances = 10
    output_dir = os.path.dirname(os.path.abspath(__file__))
    for n_queens in n_queens_list:
        for number_objs in number_objs_list:
            for instance_number in range(number_instances):
                data_to_write = generate_score_board_m_obj(n_queens, number_objs)
                write_score_board_list(data_to_write, output_dir, instance_number + 1)
