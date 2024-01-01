package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;


import java.util.ArrayList;

/**
 * a_stack
 * 
 * Rule 405.2: The stack keeps track of the order that spells and/or abilities were added to [the stack]. Each time an object in put on the stack, the object is put on top of all objects already there.
 */
public class a_stack {

	private ArrayList<Object> List_Of_Spells_Or_Nonmana_Activated_Abilities;
	
	public a_stack() {
		this.List_Of_Spells_Or_Nonmana_Activated_Abilities = new ArrayList<Object>();
	}
	
	public boolean contains_objects() {
		if (this.List_Of_Spells_Or_Nonmana_Activated_Abilities.size() > 0) {
			return true;
		}
		else {
			return false;
		}
	}
	
	public Object provides_its_top_object() {
		return this.List_Of_Spells_Or_Nonmana_Activated_Abilities.remove(this.List_Of_Spells_Or_Nonmana_Activated_Abilities.size() - 1);
	}
	
	public void receives(Object The_Object) {
		if ((The_Object instanceof a_spell) || (The_Object instanceof a_nonmana_activated_ability)) {
			this.List_Of_Spells_Or_Nonmana_Activated_Abilities.add(The_Object);
		}
	}
	
	@Override
	public String toString() {
		StringBuilder The_String_Builder = new StringBuilder("Stack: ");
		for (int i = 0; i < this.List_Of_Spells_Or_Nonmana_Activated_Abilities.size() - 1; i++) {
			The_String_Builder.append(this.List_Of_Spells_Or_Nonmana_Activated_Abilities.get(i) + "; ");
		}
		if (this.List_Of_Spells_Or_Nonmana_Activated_Abilities.size() > 0) {
			The_String_Builder.append(this.List_Of_Spells_Or_Nonmana_Activated_Abilities.get(this.List_Of_Spells_Or_Nonmana_Activated_Abilities.size() - 1));
		}
		return The_String_Builder.toString();
	}
	
}
