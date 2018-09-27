import React from "react";
import { createDrawerNavigator } from "react-navigation";
import HomeScreen from "./screens/HomeScreen";
import CameraScreen from "./screens/CameraScreen";

const Drawer = createDrawerNavigator({
  Home: { screen: HomeScreen },
  Camera: { screen: CameraScreen }
});

export default class App extends React.Component {

  render = () => {
    return <Drawer />;
  };

}
