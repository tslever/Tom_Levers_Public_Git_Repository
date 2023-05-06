package settler_utilities;


import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import lattice_utilities.World;


class SettlersTest {

	@Test
	void testImprint() {
		
		int heightOfWorld = 2;
		
		int widthOfWorld = 2;
		
		World world = new World(heightOfWorld, widthOfWorld);
		
		Settlers settlers = Settlers.organizeBasedOn(world, heightOfWorld * widthOfWorld);
		
		world = settlers.settleIndividualCellsOf(world);
		
		world.display();
		
	}

}
