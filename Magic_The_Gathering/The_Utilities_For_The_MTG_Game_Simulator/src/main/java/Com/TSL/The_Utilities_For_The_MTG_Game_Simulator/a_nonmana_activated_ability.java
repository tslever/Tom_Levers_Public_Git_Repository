package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

import java.util.ArrayList;

public class a_nonmana_activated_ability extends an_activated_ability {

	private ArrayList<ArrayList<a_mana_ability>> List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This;
	
	public a_nonmana_activated_ability(String The_Cost_To_Use, String The_Effect_To_Use, a_permanent The_Permanent_With_This_Nonmana_Activated_Ability) {
		super(The_Cost_To_Use, The_Effect_To_Use, The_Permanent_With_This_Nonmana_Activated_Ability);
	}
	
	public void activates() {
		this.permanent().taps();
	}
	
	public ArrayList<ArrayList<a_mana_ability>> list_of_combinations_of_available_mana_abilities_sufficient_to_play_this() {
		return this.List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This;
	}
	
	public a_mana_cost mana_cost() {
		if (this.cost().equals("T")) {
			return new a_mana_cost(0, 0, 0, 0, 0, 0);
		} else {
			return new a_mana_cost(0, 0, 0, 0, 0, 0);
		}
	}
	
	public void receives(ArrayList<ArrayList<a_mana_ability>> The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This_To_Use) {
		this.List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This = The_List_Of_Combinations_Of_Available_Mana_Abilities_Sufficient_To_Play_This_To_Use;
	}
	
}
