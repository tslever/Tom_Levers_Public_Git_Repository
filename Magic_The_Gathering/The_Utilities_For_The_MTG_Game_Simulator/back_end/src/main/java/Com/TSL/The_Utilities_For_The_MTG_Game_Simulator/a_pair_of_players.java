package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.Scanner;

public class a_pair_of_players {
	
	private a_player Active_Player;
	private a_player First_Player;
	private a_player Second_Player;
	
	public a_pair_of_players(a_player The_First_Player_To_Use, a_player The_Second_Player_To_Use) {
		
		this.First_Player = The_First_Player_To_Use;
		this.Second_Player = The_Second_Player_To_Use;
	}
	
	/**
	 * decides_the_starting_player
	 * 
	 * Rule 103.1:
	 * At the start of the game, the players determine which one of them will choose who takes the first turn.
	 * In the first game of a match (including a single-game match), the players may use any mutually agreeable method (flipping a coin, rolling dice, etc.) to do so.
	 */
	public void decides_the_starting_player() {
		if (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS) {
			this.First_Player.becomes_the_starting_player();
			this.Active_Player = this.First_Player;
		} else {
			this.Second_Player.becomes_the_starting_player();
			this.Active_Player = this.Second_Player;
		}
		System.out.println(this.Active_Player.name() + " is the starting player.");
		
	}
	
	public void draws_hands() {
		this.First_Player.draws_a_hand();
		this.Second_Player.draws_a_hand();
	}
	
	/**
	 * play
	 * 
	 * Rule 103.7:
	 * The starting player takes their first turn.
	 */
	public void play() throws Exception {
		this.decides_the_starting_player();
		this.shuffles_decks();
		this.draws_hands();
		// TODO: Implement mulligan.
		//this.mulligan();
		boolean should_continue = true;
		Scanner The_Scanner = new Scanner(System.in);
		while (should_continue) {
			this.Active_Player.takes_her_turn();
			this.Active_Player = this.Active_Player.other_player();
			System.out.print("Should continue (true / false)? ");
			should_continue = The_Scanner.nextBoolean();
			System.out.println();
		}
	}
	
	/** Rule 103.2:
	 * After the starting player has been determined, each player shuffles their deck so that the cards are in a random order.
	 */
	public void shuffles_decks() {
		this.First_Player.shuffles_her_deck();
		this.Second_Player.shuffles_her_deck();
	}
}