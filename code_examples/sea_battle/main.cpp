#include <SFML/Graphics.hpp>
#include <vector>
#include "game.h"

// ���������� � ������������� ��������

// ������ � ������ (� ��������) ���� ����������
const size_t WIDTH = 1000, HEIGTH = 700;

// ���������� (� ��������) �������� ������ ���� ���� ������-������������
const size_t X1 = 50, Y1 = 150;

// ���������� (� ��������) �������� ������ ���� ���� ������-����������
const size_t X2 = 550, Y2 = 150;

// �������� (� ��������) ������� ������ ���� 10 �� 10, � ����� �������� ������� ����� ������
const size_t SIDE = 400, CELL_SIDE = 40;

// �����, ������������ ���������� ���� ����� � ���� ����������
// � �������� int main() ��������������� ����������� �������:
// MouseButtonPressed(float xf, float yf) - ������������ ���� ���� � ����� � ������������ (xf, yf)
// Draw(sf::RenderWindow& window) - ������ ��� �������, �����, ������ � ������ ������ �� ������
class GUI {
private:
    // ������ ������ Game, ���������� ��� ������ ����
    Game game;

    // ������������ � �������� �����
    sf::Font font;

    // �������, ������������ ���� ������, ���� ����������, ���������� � ����� ���� ��������������
    sf::Text player_field_label, enemy_field_label, endgame_label;

    // ������ ����� ��������� ����� ������� � ������ ���� ��������� ����� �� ���� ����� ��������������
    std::vector<sf::RectangleShape> field_grids, figures;

private:
    // ������ ����� ��������� ����� �������
    void CreateFieldGrids() {
        for (size_t i = 0; i < 11; ++i) {
            field_grids.push_back(sf::RectangleShape(sf::Vector2f(2.f, SIDE)));
            field_grids[field_grids.size() - 1].move(X1 + i * SIDE / 10, Y1);
            field_grids[field_grids.size() - 1].setFillColor(sf::Color::Blue);
        }
        for (size_t i = 0; i < 11; ++i) {
            field_grids.push_back(sf::RectangleShape(sf::Vector2f(2.f, SIDE)));
            field_grids[field_grids.size() - 1].move(X2 + i * SIDE / 10, Y2);
            field_grids[field_grids.size() - 1].setFillColor(sf::Color::Blue);
        }

        for (size_t i = 0; i < 11; ++i) {
            field_grids.push_back(sf::RectangleShape(sf::Vector2f(SIDE, 2.f)));
            field_grids[field_grids.size() - 1].move(X1, Y1 + i * SIDE / 10);
            field_grids[field_grids.size() - 1].setFillColor(sf::Color::Blue);
        }
        for (size_t i = 0; i < 11; ++i) {
            field_grids.push_back(sf::RectangleShape(sf::Vector2f(SIDE, 2.f)));
            field_grids[field_grids.size() - 1].move(X2, Y2 + i * SIDE / 10);
            field_grids[field_grids.size() - 1].setFillColor(sf::Color::Blue);
        }
    }

    // ������ ��������� � ������ (i, j) ����, ������� ����� ���� �������� ����� ���������� (x, y) � ��������
    void AddTimes(size_t x, size_t y, size_t i, size_t j) {
        float width = 4;
        figures.push_back(sf::RectangleShape(sf::Vector2f(SIDE / 10, width)));
        figures[figures.size() - 1].rotate(45.f);
        figures[figures.size() - 1].move(x + (j - 1) * float(SIDE) / 10 + width * 2.1, y + (i - 1) * float(SIDE) / 10 + width * 1.7);
        figures[figures.size() - 1].setFillColor(sf::Color::Red);

        figures.push_back(sf::RectangleShape(sf::Vector2f(SIDE / 10, width)));
        figures[figures.size() - 1].rotate(-45.f);
        figures[figures.size() - 1].move(x + (j - 1) * float(SIDE) / 10 + width * 1.7, y + (i - 1) * float(SIDE) / 10 + width * 8.6);
        figures[figures.size() - 1].setFillColor(sf::Color::Red);
    }
    
    // ������ ������ � ������ (i, j) ����, ������� ����� ���� �������� ����� ���������� (x, y) � ��������
    void AddMark(size_t x, size_t y, size_t i, size_t j) {
        float width = 12;
        figures.push_back(sf::RectangleShape(sf::Vector2f(width, width)));
        figures[figures.size() - 1].move(x + (j - 1) * SIDE / 10 + width + 3, y + (i - 1) * SIDE / 10 + width + 3);
        figures[figures.size() - 1].setFillColor(sf::Color(0, 0, 0, 0));
        figures[figures.size() - 1].setOutlineColor(sf::Color(255, 0, 0, 100));
        figures[figures.size() - 1].setOutlineThickness(2);
    }

    // ������ ������ ���������� ������� � ������ (i, j) ����, 
    // ������� ����� ���� �������� ����� ���������� (x, y) � ��������
    void AddShipCell(size_t x, size_t y, size_t i, size_t j) {
        figures.push_back(sf::RectangleShape(sf::Vector2f(CELL_SIDE, CELL_SIDE)));
        figures[figures.size() - 1].move(x + (j - 1) * CELL_SIDE, y + (i - 1) * CELL_SIDE);
        figures[figures.size() - 1].setFillColor(sf::Color(100, 100, 100, 200));
    }

    // ����� ������� ���� ������� � ������ ������� ������ figures ���� ����� �� �����
    // ��� �� ���������� ���������
    void RecreateFiguresAfterTurn() {
        // �������� figures
        figures.clear();
        
        // ��������� ���� ����� �� ���� ������-������������
        const std::vector<std::vector<int>>& player_field = game.GetPlayerField();
        for (size_t i = 1; i <= 10; ++i) {
            for (size_t j = 1; j <= 10; ++j) {
                // ������ �� ������ ��������
                if (player_field[i][j] == 10 || player_field[i][j] == 11) {
                    AddShipCell(X1, Y1, i, j);
                }

                // �������� �� �������-������ � ������ (i, j)
                if (player_field[i][j] == 1 || player_field[i][j] == -2) {
                    AddMark(X1, Y1, i, j);
                }

                // �������� �� �������-��������� � ������ (i, j)
                if (player_field[i][j] == 11) {
                    AddTimes(X1, Y1, i, j);
                }
            }
        }

        // ��������� ���� ����� �� ���� ������-����������
        const std::vector<std::vector<int>>& enemy_field = game.GetEnemyField();
        for (size_t i = 1; i <= 10; ++i) {
            for (size_t j = 1; j <= 10; ++j) {
                // �������� �� �������-������ � ������ (i, j)
                if (enemy_field[i][j] == 1 || enemy_field[i][j] == -2) {
                    AddMark(X2, Y2, i, j);
                }

                // �������� �� �������-��������� � ������ (i, j)
                if (enemy_field[i][j] == 11) {
                    AddTimes(X2, Y2, i, j);
                }
            }
        }
    }
    
public:
    // ����������� ������������ ����������
    GUI() {
        // �������� ������
        font.loadFromFile("Micra Normal.ttf");
        
        // ��������� ���������� ������� "���� ������"
        player_field_label.setString("Your field");
        player_field_label.setFont(font);
        player_field_label.setCharacterSize(30);
        player_field_label.setFillColor(sf::Color::Black);
        player_field_label.setPosition(100, 50);

        // ��������� ���������� ������� "���� ����������"
        enemy_field_label.setString("Enemy field");
        enemy_field_label.setFont(font);
        enemy_field_label.setCharacterSize(30);
        enemy_field_label.setFillColor(sf::Color::Black);
        enemy_field_label.setPosition(600, 50);

        // ��������� ���������� ������� �� ��������� ����
        endgame_label.setFont(font);
        endgame_label.setCharacterSize(30);
        endgame_label.setFillColor(sf::Color::Black);

        // �������� ����� (� ������ �� �������)
        CreateFieldGrids();

        // ���������� ��������� ����������� �������� �� ����� �������
        game.GenerateRandomShipsArrangement(true);
        game.GenerateRandomShipsArrangement(false);

        // ������������ ������� � ��������� �������
        RecreateFiguresAfterTurn();
    }

    // ������������ ���� ���� � ����� � ������������ (xf, yf)
    void MouseButtonPressed(float xf, float yf) {
        // �� ����������� �� ����
        if (game.IsFinished()) {
            return;
        }

        size_t xc, yc;
        xc = round(xf);
        yc = round(yf);

        // ��������, ���������� �� ������ ���� �� ���� ������-����������
        if (X2 <= xc && xc <= X2 + SIDE && Y2 <= yc && yc <= Y2 + SIDE) {
            size_t i, j;
            j = (xc - X2) / (SIDE / 10) + 1;
            i = (yc - Y2) / (SIDE / 10) + 1;
            game.MouseButtonPressed(i, j);
            RecreateFiguresAfterTurn();

            // ����� ������� ���� ��������: �� ����������� �� ����
            if (game.UserHasWon()) {
                endgame_label.setString("Congratulations! You've won!");
                endgame_label.setPosition(160, 600);
            } else if (game.ComputerHasWon()) {
                endgame_label.setString("You've lost...");
                endgame_label.setPosition(370, 600);
            }
        }
    }

    // ������ ��� �������, �����, ������ � ������ ������ � ���� window
    void Draw(sf::RenderWindow& window) {
        for (const sf::RectangleShape& line : field_grids) {
            window.draw(line);
        }

        for (const sf::RectangleShape& fig : figures) {
            window.draw(fig);
        }

        window.draw(player_field_label);
        window.draw(enemy_field_label);
        if (game.IsFinished()) {
            window.draw(endgame_label);
        }
    }
};

int main()
{
    // �������� ���� ����������
    sf::RenderWindow window(sf::VideoMode(WIDTH, HEIGTH), "Sea Battle");
    
    // ������ GUI, ������������ ����������
    GUI gui;

    // ���� ���� �������
    while (window.isOpen())
    {
        // ������� �������
        sf::Event event;
        while (window.pollEvent(event))
        {
            // �������� ���� �������
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::MouseButtonPressed) {
                gui.MouseButtonPressed(event.mouseButton.x, event.mouseButton.y);
            }
        }

        // �������� ����� � ������� ����� ���
        window.clear(sf::Color::White);
        // ��������� ���� ���������
        gui.Draw(window);
        // ����� ����
        window.display();
    }

    return 0;
}