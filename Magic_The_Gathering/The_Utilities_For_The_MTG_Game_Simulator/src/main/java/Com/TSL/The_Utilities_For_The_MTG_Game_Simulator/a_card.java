package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

/**
 * a_card
 *
 * Rule 200.1: The parts of a card are[ though not all cards have all parts ]name, mana cost, illustration, color indicator, type line, expansion symbol, text box, power and toughness, loyalty, hand modifier, life modifier, illustration credit, legal text, and collection number. Some cards may have more than one of any or all of these parts.
 * Rule 207.1: The text box is printed on the lower half of the card. [The text box] usually contains rules text defining the card's abilities.
 */
public abstract class a_card {

	//private String Expansion;
	private boolean Is_Playable;
	private String Name;
	private ArrayList<String> Text;
	private String Type;
	
	protected a_card(String The_Expansion_To_Use, String The_Name_To_Use, ArrayList<String> The_Text_To_Use, String The_Type_To_Use) {
		//this.Expansion = The_Expansion_To_Use;
		this.Is_Playable = false;
		this.Name = The_Name_To_Use;
		this.Text = The_Text_To_Use;
		this.Type = The_Type_To_Use;
	}

	public void becomes_not_playable() {
		this.Is_Playable = false;
	}
	
	public void becomes_playable() {
		this.Is_Playable = true;
	}
	
	public boolean is_playable() {
		return this.Is_Playable;
	}
	
	public String name() {
		return this.Name;
	}
	
	public ArrayList<String> text() {
		return this.Text;
	}
	
	public String type() {
		return this.Type;
	}
	
	@Override
	public String toString() {
		return this.Name;
	}
}
