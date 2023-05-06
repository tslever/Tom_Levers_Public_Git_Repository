package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;


public class a_stack {

	private ArrayList<a_spell> List_Of_Spells;
	
	public a_stack() {
		this.List_Of_Spells = new ArrayList<a_spell>();
	}
	
	public boolean contains_spells() {
		if (this.List_Of_Spells.size() > 0) {
			return true;
		}
		else {
			return false;
		}
	}
	
	public a_spell provides_its_top_spell() {
		return this.List_Of_Spells.remove(this.List_Of_Spells.size() - 1);
	}
	
	public void receives(a_spell The_Spell) {
		this.List_Of_Spells.add(The_Spell);
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("The Stack");
		for (a_spell The_Spell : this.List_Of_Spells) {
			The_String_Builder.append("\n");
			The_String_Builder.append(The_Spell);
		}
		return The_String_Builder.toString();
	}
	
}
