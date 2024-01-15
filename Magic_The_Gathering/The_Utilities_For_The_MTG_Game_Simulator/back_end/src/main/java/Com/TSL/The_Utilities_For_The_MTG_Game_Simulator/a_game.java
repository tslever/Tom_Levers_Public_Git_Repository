package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_game {
	
	private a_player Active_Player;
	private a_deck_builder Deck_Builder;
	private a_player First_Player;
	private String Next_Action;
	private a_player Second_Player;
	private a_stack Stack;
	private String Summary_Of_Action;
	
	public a_game() {
		this.Next_Action = "Set Up First Player";
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
		this.Summary_Of_Action = this.Active_Player.name() + " is the starting player.";
		System.out.println(this.Summary_Of_Action);
		return this.Summary_Of_Action;
	}
	
	public void draws_hands() {
		this.First_Player.draws_a_hand();
		this.Second_Player.draws_a_hand();
	}
	
	public String begins_setting_up_the_first_player() {
    	this.Stack = new a_stack();
    	this.Deck_Builder = new a_deck_builder();
    	a_deck The_Deck_Keep_The_Peace = this.Deck_Builder.builds_Keep_The_Peace();
		this.First_Player = new a_player(The_Deck_Keep_The_Peace, this, "Tom", this.Stack);
		this.Next_Action = "Set Up Second Player";
		this.Summary_Of_Action = "First player " + this.First_Player + " will play the following deck of " + The_Deck_Keep_The_Peace.number_of_cards() + " cards. " + The_Deck_Keep_The_Peace;
		System.out.println(this.Summary_Of_Action);
		return this.Summary_Of_Action;
	}
	
	public String completes_setting_up_both_players() {
    	a_deck The_Deck_Large_And_In_Charge = this.Deck_Builder.builds_Large_And_In_Charge();
    	this.Second_Player = new a_player(The_Deck_Large_And_In_Charge, this, "Scott", this.Stack);
    	this.First_Player.receives(this.Second_Player);
    	this.Second_Player.receives(this.First_Player);
    	this.Next_Action = "Decide Starting Player";
    	this.Summary_Of_Action = "Second player " + this.Second_Player + " will play the following deck of " + The_Deck_Large_And_In_Charge.number_of_cards() + " cards. " + The_Deck_Large_And_In_Charge;
    	System.out.println(this.Summary_Of_Action);
    	return this.Summary_Of_Action;
	}

	public String advances() {
		if (this.Next_Action.equals("Set Up First Player")) {
			return this.begins_setting_up_the_first_player();
		} else if (this.Next_Action.equals("Set Up Second Player")) {
			return this.completes_setting_up_both_players();
		}
		else if (this.Next_Action.equals("Decide Starting Player")) {
			this.Next_Action = "Shuffle Decks";
			return this.decides_the_starting_player();
		} else if (this.Next_Action.equals("Shuffle Decks")) {
			this.Next_Action = "First Player Draws Hand";
			return this.shuffles_decks();
		} else if (this.Next_Action.equals("First Player Draws Hand")) {
			this.Next_Action = "Second Player Draws Hand";
			return this.First_Player.draws_a_hand();
		} else if (this.Next_Action.equals("Second Player Draws Hand")) {
			// TODO: Implement performing mulligan.
			//this.Next_Action = "First Player Mulligans";
			//this.Next_Action = "Second Player Mulligans";
			this.Next_Action = "Active Player Takes Turn";
			return this.Second_Player.draws_a_hand();
		} else {
			return this.Active_Player.takes_her_turn();
		}
	}
	
	/** Rule 103.2:
	 * After the starting player has been determined, each player shuffles their deck so that the cards are in a random order.
	 */
	public String shuffles_decks() {
		this.First_Player.shuffles_her_deck();
		this.Second_Player.shuffles_her_deck();
		this.Summary_Of_Action = "Each player has shuffled the player's deck.";
		System.out.println(this.Summary_Of_Action);
		return this.Summary_Of_Action;
	}
	
	public String next_action() {
		return this.Next_Action;
	}
	
	public void next_action_becomes(String The_Next_Action_To_Use) {
		this.Next_Action = The_Next_Action_To_Use;
	}
}