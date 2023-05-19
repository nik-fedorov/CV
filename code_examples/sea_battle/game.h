#include <cassert>
#include <ctime>
#include <random>
#include <set>
#include <utility>
#include <vector>

// �����, ���������� ��� ������ ���� (�� ���������� ���������� � ����������� �����������!)
// ������ � ���� ������� - ���� �������
// � ������� GUI ����� main.cpp ��������������� ����������� �������:
// MouseButtonPressed(size_t i, size_t j) - ��������� �������� � ������ (i, j) ���� ������-����������
// GetPlayerField() � GetEnemyField() - ���������� �������� ��������� ����������� ������ �� 
// �������������� ���� ������� ��� �� ���������� ���������
// IsFinished(), UserHasWon(), ComputerHasWon() - �������� �� ��������� ���� � �� ��, ��� �������
// GenerateRandomShipsArrangement(bool for_player) - ���������� ��������� ����������� �������� �� ����
// ��������� ������ ����� ������ ������ Game ��� ��� ���������� ������, ���� ��� ������������ ������ Game
class Game {
private:
    // ���������� ���������� �� ����� �������� � ������������ � ���������� ��������������
    size_t player_counter, enemy_counter;

    /* ���� �������
    ����������� ����� � ������� �����:
    0 - ������ ������
    1 - ������ ������, � ������� ������������ �������
    10 - ������, ������� ��������
    11 - ������, ������� ��������, � ������� ������������ �������
    -1 - ������ ������, ����� � ������� ���� �������
    -2 - ������ ������, ����� � ������� ���� �������, � � ������� ������������ ������� */
    std::vector<std::vector<int>> player_field, enemy_field;

    // ��������� �������������� ������ �������� ������������, ������� ������������� ����� �
    // ��� ������������� �������� �������� ������������
    std::set<std::pair<size_t, size_t>> cells_to_shoot;

    std::mt19937 mersenne;  // ��������� ��������� �����

private:
    // �������� ����������� ������� ������� 
    // (��������� �������� ������� ������ ������������ ������� ��������� -2)
    // i, j - ������ ������ � ������� ����� �� ������ ������������ �������
    // �������� ������������ ��� ����� ������, ���� for_player == true, ����� ��� ����� ����������
    void SurroundDestroyedShip(size_t i, size_t j, bool for_player) {
        // field_ptr ��������� �� �� ����, � ������� �� ������ ��������
        std::vector<std::vector<int>>* field_ptr;
        if (for_player) {
            field_ptr = &player_field;
        } else {
            field_ptr = &enemy_field;
        }
        
        (*field_ptr)[i][j] = -100;

        if ((*field_ptr)[i][j + 1] == 11) {
            SurroundDestroyedShip(i, j + 1, for_player);
        } else if ((*field_ptr)[i][j + 1] == -1) {
            (*field_ptr)[i][j + 1] = -2;
        }

        if ((*field_ptr)[i][j - 1] == 11) {
            SurroundDestroyedShip(i, j - 1, for_player);
        } else if ((*field_ptr)[i][j - 1] == -1) {
            (*field_ptr)[i][j - 1] = -2;
        }

        if ((*field_ptr)[i + 1][j] == 11) {
            SurroundDestroyedShip(i + 1, j, for_player);
        } else if ((*field_ptr)[i + 1][j] == -1) {
            (*field_ptr)[i + 1][j] = -2;
        }

        if ((*field_ptr)[i - 1][j] == 11) {
            SurroundDestroyedShip(i - 1, j, for_player);
        } else if ((*field_ptr)[i - 1][j] == -1) {
            (*field_ptr)[i - 1][j] = -2;
        }


        if ((*field_ptr)[i + 1][j + 1] == -1) {
            (*field_ptr)[i + 1][j + 1] = -2;
        }

        if ((*field_ptr)[i + 1][j - 1] == -1) {
            (*field_ptr)[i + 1][j - 1] = -2;
        }

        if ((*field_ptr)[i - 1][j + 1] == -1) {
            (*field_ptr)[i - 1][j + 1] = -2;
        }

        if ((*field_ptr)[i - 1][j - 1] == -1) {
            (*field_ptr)[i - 1][j - 1] = -2;
        }


        (*field_ptr)[i][j] = 11;
    }

    // ������ ��� �� ���������
    void MakeTurn() {
        size_t i, j;
        if (cells_to_shoot.empty()) {
            // ���� ��� ������������ �� ������������� ��������, 
            // �� �������� �������� ����� ���������� �������� (i, j)
            i = mersenne() % 10 + 1;
            j = mersenne() % 10 + 1;
            while (player_field[i][j] == 1 || player_field[i][j] == 11 || player_field[i][j] == -2) {
                i = mersenne() % 10 + 1;
                j = mersenne() % 10 + 1;
            }
        } else {
            // ���� ���� ������������ �� ������������� �������, 
            // �� �������� ���������� �������������� ������ ������ �� ���
            i = cells_to_shoot.begin()->first;
            j = cells_to_shoot.begin()->second;
            cells_to_shoot.erase(cells_to_shoot.begin());
        }

        // �������� � ������ (i, j) ���� ������-������������
        if (player_field[i][j] == 0) {
            player_field[i][j] = 1;
        } else if (player_field[i][j] == -1) {
            player_field[i][j] = -2;
        } else if (player_field[i][j] == 10) {
            player_field[i][j] = 11;
            if (IsShipDestroyed(i, j, true) && player_counter > 0) {
                --player_counter;
                SurroundDestroyedShip(i, j, true);
            } else {
                // ��������� ��������� �������������� ������ �������� ������������, � ������� ����� ����������
                if (player_field[i][j + 1] == 10) {
                    cells_to_shoot.insert(std::pair<size_t, size_t>(i, j + 1));
                }
                if (player_field[i][j - 1] == 10) {
                    cells_to_shoot.insert(std::pair<size_t, size_t>(i, j - 1));
                }
                if (player_field[i - 1][j] == 10) {
                    cells_to_shoot.insert(std::pair<size_t, size_t>(i - 1, j));
                }
                if (player_field[i + 1][j] == 10) {
                    cells_to_shoot.insert(std::pair<size_t, size_t>(i + 1, j));
                }
            }
        }
    }


public:
    Game() : mersenne(time(0)), player_counter(0), enemy_counter(0) {
        player_field.resize(12, std::vector<int>(12, 0));
        enemy_field.resize(12, std::vector<int>(12, 0));
    }

    // ������������ ������� � ���� ������-������������ � ������������ (i, j)
    void MouseButtonPressed(size_t i, size_t j) {
        if (enemy_field[i][j] == 0) {
            enemy_field[i][j] = 1;
            MakeTurn();
        } else if (enemy_field[i][j] == -1) {
            enemy_field[i][j] = -2;
            MakeTurn();
        } else if (enemy_field[i][j] == 10) {
            enemy_field[i][j] = 11;
            if (IsShipDestroyed(i, j, false) && enemy_counter > 0) {
                --enemy_counter;
                SurroundDestroyedShip(i, j, false);
            }
        }
    }

    // ���������� ����������� ������ �� ���� (��������� ������) ������-������������
    const std::vector<std::vector<int>>& GetPlayerField() const {
        return player_field;
    }

    // ���������� ����������� ������ �� ���� (��������� ������) ������-����������
    const std::vector<std::vector<int>>& GetEnemyField() const {
        return enemy_field;
    }

    // ���������, ����������� �� ����
    bool IsFinished() const {
        if (player_counter == 0 || enemy_counter == 0) {
            return true;
        } else {
            return false;
        }
    }

    // ��������: ������� �� ������������
    bool UserHasWon() const {
        if (enemy_counter == 0) {
            return true;
        } else {
            return false;
        }
    }

    // ��������: ������� �� ���������
    bool ComputerHasWon() const {
        if (player_counter == 0) {
            return true;
        } else {
            return false;
        }
    }


    // ���������, ��������� �� �������, �������� ����������� ������ (i, j)
    // ��������� �� ���� ���������� (i, j) ������ ������������� �������
    // �������� ������������ ��� ����� ������, ���� for_player == true, ����� ��� ����� ����������
    bool IsShipDestroyed(size_t i, size_t j, bool for_player) {
        // field_ptr ��������� �� �� ����, � ������� �� ������ ��������
        std::vector<std::vector<int>>* field_ptr;
        if (for_player) {
            field_ptr = &player_field;
        } else {
            field_ptr = &enemy_field;
        }

        // �������� �� ��� ������ ������������ ������� �������
        if ((*field_ptr)[i][j] != 11) {
            return false;
        }

        (*field_ptr)[i][j] = -100;

        if ((*field_ptr)[i][j + 1] == 10) {
            (*field_ptr)[i][j] = 11;
            return false;
        } else if ((*field_ptr)[i][j + 1] == 11) {
            if (!IsShipDestroyed(i, j + 1, for_player)) {
                (*field_ptr)[i][j] = 11;
                return false;
            }
        }

        if ((*field_ptr)[i][j - 1] == 10) {
            (*field_ptr)[i][j] = 11;
            return false;
        } else if ((*field_ptr)[i][j - 1] == 11) {
            if (!IsShipDestroyed(i, j - 1, for_player)) {
                (*field_ptr)[i][j] = 11;
                return false;
            }
        }

        if ((*field_ptr)[i + 1][j] == 10) {
            (*field_ptr)[i][j] = 11;
            return false;
        } else if ((*field_ptr)[i + 1][j] == 11) {
            if (!IsShipDestroyed(i + 1, j, for_player)) {
                (*field_ptr)[i][j] = 11;
                return false;
            }
        }

        if ((*field_ptr)[i - 1][j] == 10) {
            (*field_ptr)[i][j] = 11;
            return false;
        } else if ((*field_ptr)[i - 1][j] == 11) {
            if (!IsShipDestroyed(i - 1, j, for_player)) {
                (*field_ptr)[i][j] = 11;
                return false;
            }
        }

        (*field_ptr)[i][j] = 11;
        return true;
    }

    // ���������, �������� �� ���������� ������� ������� size ���, ����� ��� ������� ����� ������ ����� ���������� (i, j)
    // ���� is_horizontal == true, �� ����������� ����������� ��������� ������� �������������, ����� - �����������
    // �������� ������������ ��� ����� ������, ���� for_player == true, ����� ��� ����� ����������
    bool PossibleToPlaceShip(size_t i, size_t j, size_t size, bool is_horizontal, bool for_player) {
        // field_ptr ��������� �� �� ����, � ������� �� ������ ��������
        std::vector<std::vector<int>>* field_ptr;
        if (for_player) {
            field_ptr = &player_field;
        } else {
            field_ptr = &enemy_field;
        }

        // ����� �� ��� ���������� � ������� [1, 10]
        if (i < 1 || 10 < i || j < 1 || 10 < j) {
            return false;
        }

        if (is_horizontal) {
            if (j > 11 - size) {
                return false;
            }
            bool res = true;
            for (size_t k = 0; k < size; ++k) {
                if ((*field_ptr)[i][j + k] != 0) {
                    res = false;
                }
            }
            return res;
        } else {
            if (i > 11 - size) {
                return false;
            }
            bool res = true;
            for (size_t k = 0; k < size; ++k) {
                if ((*field_ptr)[i + k][j] != 0) {
                    res = false;
                }
            }
            return res;
        }
    }

    // ��������� ������� ������� size ���, ����� ��� ������� ����� ������ ����� ���������� (i, j)
    // ���� is_horizontal == true, �� ������� ����������� �������������, ����� - �����������
    // �������� ������������ ��� ����� ������, ���� for_player == true, ����� ��� ����� ����������
    void PlaceShip(size_t i, size_t j, size_t size, bool is_horizontal, bool for_player) {
        // field_ptr ��������� �� �� ����, � ������� �� ������ ��������
        std::vector<std::vector<int>>* field_ptr;
        if (for_player) {
            field_ptr = &player_field;
        } else {
            field_ptr = &enemy_field;
        }

        if (PossibleToPlaceShip(i, j, size, is_horizontal, for_player)) {
            if (is_horizontal) {
                // ��������� �������
                for (size_t k = 0; k < size; ++k) {
                    (*field_ptr)[i][j + k] = 10;
                }

                // ��������� ������ ������ ����
                (*field_ptr)[i][j - 1] = -1;
                (*field_ptr)[i][j + size] = -1;
                for (size_t k = 0; k < size + 2; ++k) {
                    (*field_ptr)[i - 1][j - 1 + k] = -1;
                    (*field_ptr)[i + 1][j - 1 + k] = -1;
                }
            } else {
                // ��������� �������
                for (size_t k = 0; k < size; ++k) {
                    (*field_ptr)[i + k][j] = 10;
                }

                // ��������� ������ ������ ����
                (*field_ptr)[i - 1][j] = -1;
                (*field_ptr)[i + size][j] = -1;
                for (size_t k = 0; k < size + 2; ++k) {
                    (*field_ptr)[i - 1 + k][j - 1] = -1;
                    (*field_ptr)[i - 1 + k][j + 1] = -1;
                }
            }
        }

        if (for_player) {
            ++player_counter;
        } else {
            ++enemy_counter;
        }
    }

    // ���������� ��������� ����������� �������� �� ����, ��������� 1 ��������:
    // ���� for_player == true, �� ����������� ������� ������-������������
    // ����� (for_player == false) ���������� ������� ������-����������
    void GenerateRandomShipsArrangement(bool for_player) {
        size_t i, j;

        // ��������� 4-��������
        if (mersenne() % 2 == 0) {
            // ��������� �������������
            i = mersenne() % 10 + 1;
            j = mersenne() % 7 + 1;
            PlaceShip(i, j, 4, true, for_player);
        } else {
            // ��������� �����������
            j = mersenne() % 10 + 1;
            i = mersenne() % 7 + 1;
            PlaceShip(i, j, 4, false, for_player);
        }

        // ��������� 3-���������
        for (size_t ship = 0; ship < 2; ++ship) {
            if (mersenne() % 2 == 0) {
                // ��������� �������������
                i = mersenne() % 10 + 1;
                j = mersenne() % 8 + 1;
                while (!PossibleToPlaceShip(i, j, 3, true, for_player)) {
                    i = mersenne() % 10 + 1;
                    j = mersenne() % 8 + 1;
                }
                PlaceShip(i, j, 3, true, for_player);
            } else {
                // ��������� �����������
                j = mersenne() % 10 + 1;
                i = mersenne() % 8 + 1;
                while (!PossibleToPlaceShip(i, j, 3, false, for_player)) {
                    j = mersenne() % 10 + 1;
                    i = mersenne() % 8 + 1;
                }
                PlaceShip(i, j, 3, false, for_player);
            }
        }

        // ��������� 2-���������
        for (size_t ship = 0; ship < 3; ++ship) {
            if (mersenne() % 2 == 0) {
                // ��������� �������������
                i = mersenne() % 10 + 1;
                j = mersenne() % 9 + 1;
                while (!PossibleToPlaceShip(i, j, 2, true, for_player)) {
                    i = mersenne() % 10 + 1;
                    j = mersenne() % 9 + 1;
                }
                PlaceShip(i, j, 2, true, for_player);
            } else {
                // ��������� �����������
                j = mersenne() % 10 + 1;
                i = mersenne() % 9 + 1;
                while (!PossibleToPlaceShip(i, j, 2, false, for_player)) {
                    j = mersenne() % 10 + 1;
                    i = mersenne() % 9 + 1;
                }
                PlaceShip(i, j, 2, false, for_player);
            }
        }

        // ��������� 1-���������
        for (size_t ship = 0; ship < 4; ++ship) {
            i = mersenne() % 10 + 1;
            j = mersenne() % 10 + 1;
            while (!PossibleToPlaceShip(i, j, 1, true, for_player)) {
                i = mersenne() % 10 + 1;
                j = mersenne() % 10 + 1;
            }
            PlaceShip(i, j, 1, true, for_player);
        }
    }

    
    // ��������� ������� ����� ������ ��� ������������ � �� ��� ���� ������
    // ���������� ��������, ���������� � ������ (i, j) ���� ������-������������, ���� for_player == true, 
    // ����� - ���� ������-����������
    int GetCellValue(size_t i, size_t j, bool for_player) const {
        assert(i >= 1 && i <= 10 && j >= 1 && j <= 10);  // ����� �� ��� ���������� � ������� [1, 10]
        
        if (for_player) {
            return player_field[i][j];
        } else {
            return enemy_field[i][j];
        }
    }
};