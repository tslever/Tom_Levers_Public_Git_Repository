package com.tsl.turngenerator;

import java.io.IOException;

public class TurnGenerator {
	
	public static void main(String[] args) throws IOException {
		
		Board board = new Board();
		
		Player player0 = new Player(0);
		player0.activate();
		Player player1 = new Player(1);
		Player player2 = new Player(2);
		Player player3 = new Player(3);
		Player[] players = new Player[] {player0, player1, player2, player3};
		
		Turn turn = new Turn(board, new Bank(), players);
		turn.write("resources/testImage.PNG");

	}
}
