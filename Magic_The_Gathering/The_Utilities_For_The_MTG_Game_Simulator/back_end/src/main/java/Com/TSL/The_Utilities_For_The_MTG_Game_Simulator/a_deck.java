package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;
import java.util.Collections;

public class a_deck {
	
	private ArrayList<a_card> List_Of_Cards;
	private String Name;
	
	
	public a_deck(ArrayList<a_card> The_List_Of_Cards_To_Use, String The_Name_To_Use) {
		this.List_Of_Cards = The_List_Of_Cards_To_Use;
		this.Name = The_Name_To_Use;
	}
		
	public a_card removes_and_provides_its_top_card() {
		return this.List_Of_Cards.remove(this.List_Of_Cards.size() - 1);
	}
	
	public String name() {
		return this.Name;
	}
	
	public int number_of_cards() {
		return this.List_Of_Cards.size();
	}
	
	public void shuffles() {
		Collections.shuffle(List_Of_Cards);
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder(this.Name + ": ");
		for (int i = 0; i < this.List_Of_Cards.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Cards.get(i));
			The_String_Builder.append("; ");
		}
		The_String_Builder.append(this.List_Of_Cards.get(this.List_Of_Cards.size() - 1));
		return The_String_Builder.toString();
	}
}