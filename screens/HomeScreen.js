/* jshint esversion: 6 */

import React from "react";
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
			<View style={styles.container}>
				<Text>Welcome to the home screen.</Text>
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

const white = "white";
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: white,
    alignItems: "center",
    justifyContent: "center",
  }
});