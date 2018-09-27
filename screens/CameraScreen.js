import React from "react";
import { Styles } from "../configs/Styles";
import {
  StyleSheet,
  Text,
  View,
  Button
} from "react-native";

export default class CameraScreen extends React.Component {

	static navigationOptions = {
		drawerLabel: "Camera"
	};

	render = () => {
		return (
			<View style={Styles.container}>
				<Text>Welcome to the camera screen. OpenCV and camera coming soon!</Text>
				<Button
				title="Go to home screen"
				onPress={ () => this.props.navigation.navigate("Home") }
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