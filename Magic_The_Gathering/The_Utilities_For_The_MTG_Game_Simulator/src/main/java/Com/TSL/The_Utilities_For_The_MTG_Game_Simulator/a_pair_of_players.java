package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;



public class a_pair_of_players {
	
	private a_player Active_Player;
	private a_player First_Player;
	private a_player Second_Player;
	
	
	public a_pair_of_players(a_player The_First_Player_To_Use, a_player The_Second_Player_To_Use) {
		
		this.First_Player = The_First_Player_To_Use;
		this.Second_Player = The_Second_Player_To_Use;
	}
	
	
	/** Rule 103.1:
	 * At the start of the game, the players determine which one of them will choose who takes the first turn. In the first game of a match (including a
	 * single-game match), the players may use any mutually agreeable method (flipping a coin, rolling dice, etc.) to do so.
	 */
	
	public void decide_the_starting_player() {
		
		if (a_coin_flip.provides_a_random_value() == a_coin_flip.HEADS) {
			this.First_Player.becomes_the_starting_player();
			this.Active_Player = this.First_Player;
		}
		else {
			this.Second_Player.becomes_the_starting_player();
			this.Active_Player = this.Second_Player;
		}
		System.out.println(this.Active_Player.provides_her_name() + " is the starting player.\n");
		
	}
	
	
	public void draw_hands() {
		this.First_Player.draws_a_hand();
		this.Second_Player.draws_a_hand();
	}
	
	
	/** Rule 103.7:
	 * The starting player takes their first turn.
	 */
	
	public void play() {
		this.decide_the_starting_player();
		this.shuffle_their_decks();
		this.draw_hands();
		// TODO: Implement mulligan.
		//this.mulligan();
		this.Active_Player.takes_her_turn();
	}
	
	
	/** Rule 103.2:
	 * After the starting player has been determined, each player shuffles their deck so that the cards are in a random order.
	 */
	
	public void shuffle_their_decks() {
		this.First_Player.shuffles_her_deck();
		this.Second_Player.shuffles_her_deck();
	}

}