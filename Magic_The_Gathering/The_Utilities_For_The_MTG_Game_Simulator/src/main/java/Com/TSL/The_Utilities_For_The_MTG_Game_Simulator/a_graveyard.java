package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_graveyard {

	private ArrayList<a_card> List_Of_Cards;
	
	public a_graveyard() {
		this.List_Of_Cards = new ArrayList<a_card>();
	}
	
	public void receives(a_card The_Card) {
		this.List_Of_Cards.add(The_Card);
	}
}
