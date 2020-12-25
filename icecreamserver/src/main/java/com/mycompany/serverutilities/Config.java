/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities;

/**
 *
 * @author thoma
 */
public class Config {
    
    public Config() {
        
    }
    
    public MessageInterface[] getMessageInterfaces() {
        
        MessageInterface one = new MessageInterface(
                "/one",
                (InterfaceForProcessing) () -> { System.out.println("/one"); });
        
        MessageInterface two = new MessageInterface(
            "/two",
            (InterfaceForProcessing) () -> { System.out.println("/two"); });
        
        return new MessageInterface[]{one, two};
        
    }
    
}
