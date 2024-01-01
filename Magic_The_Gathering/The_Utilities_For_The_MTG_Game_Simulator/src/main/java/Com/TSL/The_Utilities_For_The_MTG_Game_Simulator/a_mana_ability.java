package Com.TSL.The_Utilities_For_The_MTG_Game_Simulator;

public class a_mana_ability extends an_activated_ability {

	public a_mana_ability(String The_Cost_To_Use, String The_Effect_To_Use, a_permanent The_Permanent_With_This_Mana_Ability) {
		super(The_Cost_To_Use, The_Effect_To_Use, The_Permanent_With_This_Mana_Ability);
	}
	
	public a_mana_pool activates_and_contributes_a_mana_pool() {
		if (this.provides_its_effect().equals("Add [G].")) {
			if (this.provides_its_cost().equals("T")) {
				this.provides_its_permanent().taps();
			}
			return new a_mana_pool(0, 0, 0, 1, 0, 0);
		} else if (this.provides_its_effect().equals("Add [W].")) {
			if (this.provides_its_cost().equals("T")) {
				this.provides_its_permanent().taps();
			}
			return new a_mana_pool(0, 0, 0, 0, 0, 1);
		} else {
			return new a_mana_pool(0, 0, 0, 0, 0, 0);
		}
	}
	
	public a_mana_pool indicates_the_mana_pool_it_would_contribute() {
		if (this.provides_its_effect().equals("Add [G].")) {
			return new a_mana_pool(0, 0, 0, 1, 0, 0);
		} else if (this.provides_its_effect().equals("Add [W].")) {
			return new a_mana_pool(0, 0, 0, 0, 0, 1);
		} else {
			return new a_mana_pool(0, 0, 0, 0, 0, 0);
		}
	}	
	
}
