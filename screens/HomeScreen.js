/* jshint esversion: 6 */

import React from "react";
import { Styles } from "../configs/Styles";
import { 
  StyleSheet,
  Text,
  View,
  Button
} from "react-native";

export default class HomeScreen extends React.Component {

	static navigationOptions = {
		drawerLabel: "Home"
	};

	render = () => {
		return (
			<View style={Styles.container}>
				<Text>Welcome home!</Text>
				<Button
				title="Go to camera screen"
				onPress={ () => this.props.navigation.navigate("Camera") }
				/>
				<Button
				title="Toggle drawer"
				onPress={ () => this.props.navigation.toggleDrawer() }
				/>
			</View>
		);
	};

}

const styles = StyleSheet.create({
  
});