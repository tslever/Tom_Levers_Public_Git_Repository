
// Allows executable to find class Controller.
package com.mycompany.serverutilities.clientcommunicationutilities;

/**
 * Defines class Controller that defines basic functionality to process a
 * an HTTP message.
 * @version 0.0
 * @author Tom Lever
 */
public class Controller {
    
    /**
     * Defines constructor Controller.
     */
    public Controller() {
        System.out.println("Controller constructor: Started.");
    }
    
    /**
     * Defines method process, which must be overridden to process
     * IceCreamApplicationMessages into Products.
     * @param messageToProcess
     * @throws Exception
     */
    public void process(IceCreamApplicationMessage messageToProcess)
    throws Exception {
        throw new Exception(
            "BasicController must be extended; process must be overridden.");
        // TODO: Throw BasicControllerMustBeExtendedException instead of
        // Exception.
    }
}