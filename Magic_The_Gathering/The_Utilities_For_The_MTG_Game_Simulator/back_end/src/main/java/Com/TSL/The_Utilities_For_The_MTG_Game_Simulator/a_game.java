package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.Scanner;

public class a_game {
	
	private a_player Active_Player;
	private a_player First_Player;
	private a_player Second_Player;
	private String Next_Action;
	
	public a_game() {
    	a_stack The_Stack = new a_stack();
    	a_player The_First_Player = new a_player("Tom", The_Stack);
    	a_deck_builder The_Deck_Builder = new a_deck_builder();
    	a_deck The_Deck_Keep_The_Peace = The_Deck_Builder.builds_Keep_The_Peace();
    	//a_deck_history The_Deck_History_For_Keep_the_Peace = new a_deck_history(The_Deck_Keep_The_Peace, 0, 0);
    	//System.out.println(The_Deck_History_For_Keep_the_Peace);
    	The_First_Player.receives(The_Deck_Keep_The_Peace);
    	a_player The_Second_Player = new a_player("Scott", The_Stack);
    	a_deck The_Deck_Large_And_In_Charge = The_Deck_Builder.builds_Large_And_In_Charge();
    	//a_deck_history The_Deck_History_For_Large_and_in_Charge = new a_deck_history(The_Deck_Large_And_In_Charge, 0, 0);
    	//System.out.println(The_Deck_History_For_Large_and_in_Charge);
    	The_Second_Player.receives(The_Deck_Large_And_In_Charge);
    	The_First_Player.receives(The_Second_Player);
    	The_Second_Player.receives(The_First_Player);
		this.First_Player = The_First_Player;
		this.Second_Player = The_Second_Player;
		this.Next_Action = "Decide Starting Player";
	}
	
	/**
	 * Rule 103.1:
	 * At the start of the game, the players determine which one of them will choose who takes the first turn.
	 * In the first game of a match (including a single-game match), the players may use any mutually agreeable method (flipping a coin, rolling dice, etc.) to do so.
	 */
	public String decides_the_starting_player() {
		if (an_enumeration_of_states_of_a_coin.provides_a_state() == an_enumeration_of_states_of_a_coin.HEADS) {
			this.First_Player.becomes_the_starting_player();
			this.Active_Player = this.First_Player;
		} else {
			this.Second_Player.becomes_the_starting_player();
			this.Active_Player = this.Second_Player;
		}
		String The_Summary_Of_Deciding_The_Starting_Player = this.Active_Player.name() + " is the starting player.";
		System.out.println(The_Summary_Of_Deciding_The_Starting_Player);
		return The_Summary_Of_Deciding_The_Starting_Player;
	}
	
	public void draws_hands() {
		this.First_Player.draws_a_hand();
		this.Second_Player.draws_a_hand();
	}

	public String advances() {
		if (this.Next_Action.equals("Decide Starting Player")) {
			this.Next_Action = "Shuffle Decks";
			return this.decides_the_starting_player();
		} else {
			return "TODO";
		}
//		this.shuffles_decks();
//		this.draws_hands();
//		// TODO: Implement mulligan.
//		//this.mulligan();
//		boolean should_continue = true;
//		Scanner The_Scanner = new Scanner(System.in);
//		while (should_continue) {
		    // Rule 103.7: The starting player takes their first turn.
//			this.Active_Player.takes_her_turn();
//			this.Active_Player = this.Active_Player.other_player();
//			System.out.print("Should continue (true / false)? ");
//			should_continue = The_Scanner.nextBoolean();
//			System.out.println();
//		}
	}
	
	/** Rule 103.2:
	 * After the starting player has been determined, each player shuffles their deck so that the cards are in a random order.
	 */
	public void shuffles_decks() {
		this.First_Player.shuffles_her_deck();
		this.Second_Player.shuffles_her_deck();
	}
}