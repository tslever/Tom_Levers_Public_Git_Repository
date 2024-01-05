package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_deck_history {
	
	private a_deck Deck;
	private int Number_Of_Losses;
	private int Number_Of_Wins;
	
	public a_deck_history(a_deck The_Deck_To_Use, int The_Number_Of_Losses_To_Use, int The_Number_Of_Wins_To_Use) {
		this.Deck = The_Deck_To_Use;
		this.Number_Of_Losses = The_Number_Of_Losses_To_Use;
		this.Number_Of_Wins = The_Number_Of_Wins_To_Use;
	}
	
	@Override
	public String toString() {
		return "The deck " + this.Deck.provides_its_name() + " has " + this.Number_Of_Losses + " losses and " + this.Number_Of_Wins + " wins.";
	}
}
