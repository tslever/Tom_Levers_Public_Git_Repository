package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;

/**
 * a_stack
 * 
 * Rule 405.2: The stack keeps track of the order that spells and/or abilities were added to [the stack]. Each time an object in put on the stack, the object is put on top of all objects already there.
 */
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
		StringBuilder The_String_Builder = new StringBuilder("Stack: ");
		for (int i = 0; i < this.List_Of_Spells.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Spells.get(i) + "; ");
		}
		if (this.List_Of_Spells.size() > 0) {
			The_String_Builder.append(this.List_Of_Spells.get(this.List_Of_Spells.size() - 1));
		}
		return The_String_Builder.toString();
	}
	
}
