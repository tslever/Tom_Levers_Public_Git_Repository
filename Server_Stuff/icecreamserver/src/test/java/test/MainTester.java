/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;


import com.mycompany.serverutilities.clientcommunicationutilities.SetMessageInterfacesException;
import com.mycompany.serverutilities.clientcommunicationutilities.StartServerListeningForMessagesException;
import com.mycompany.serverutilities.InvalidPortNumberException;
import com.mycompany.serverutilities.Main;
import java.io.IOException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


/**
 *
 * @author thoma
 */
public class MainTester {
    
    @Test
    public void testMain()
        throws InvalidPortNumberException,
               IOException,
               SetMessageInterfacesException,
               StartServerListeningForMessagesException {
        
        Main instanceOfClassMain = new Main();
        
        instanceOfClassMain.main( new String[]{"4444"} );
        
        Assertions.assertTrue(
            instanceOfClassMain.getMessageInterfacesHaveBeenSet());
        
    }
    
}
