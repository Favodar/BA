from gazebo_msgs.srv import GetModelState
import rospy
import math

class Block:
    def __init__(self, name, relative_entity_name):
        self._name = name
        self._relative_entity_name = relative_entity_name

class Tutorial:

    _blockListDict = {
        'block_a': Block('panda', 'panda_rightfinger'),
        'block_b': Block('panda', 'panda_link0'),
        # 'block_b': Block('panda', 'panda_leftfinger'),
        # 'block_b': Block('unit_sphere', 'link'),

    }

    def show_gazebo_models(self):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            for block in self._blockListDict:
                blockName = str(block._name)
                resp_coordinates = model_coordinates(blockName, block._relative_entity_name)
                print ('\n')
                print ('Status.success = ' + resp_coordinates.success)
                print(blockName)
                print("Cube " + str(block._name))
                print("Valeur de X : " + str(resp_coordinates.pose.position.x))
                print("Valeur de Y : " + str(resp_coordinates.pose.position.y))

                print("Valeur de " + self._blockListDict.get('block_a')._name + self._blockListDict.get('block_a')._relative_entity_name)
                print("Valeur de " + block._name + block._relative_entity_name)

        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))

    def distance(self):
        try:
            model_coordinates = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
            
            blockName1 = 'block_a'
            blockName2 = 'block_b'

            coordinates1 = model_coordinates(self._blockListDict.get('block_a')._name, self._blockListDict.get('block_a')._relative_entity_name)
            coordinates2 = model_coordinates(self._blockListDict.get('block_b')._name, self._blockListDict.get('block_b')._relative_entity_name)
            #print 'coordinates1.success = ', coordinates1.success
            #print 'coordinates2.success = ', coordinates2.success
            xdistance = coordinates1.pose.position.x - coordinates2.pose.position.x
            ydistance = coordinates1.pose.position.y - coordinates2.pose.position.y
            zdistance = coordinates1.pose.position.z - coordinates2.pose.position.z
            distance = math.sqrt(xdistance**2 + ydistance**2 + zdistance**2)
            # print str(xdistance) + " " + str(ydistance)
            #print distance
            return distance

            # for block in self._blockListDict.itervalues():
            #     blockName = str(block._name)
            #     resp_coordinates = model_coordinates(blockName, block._relative_entity_name)
            #     print '\n'
            #     print 'Status.success = ', resp_coordinates.success
            #     print(blockName)
            #     print("Cube " + str(block._name))
            #     print("Valeur de X : " + str(resp_coordinates.pose.position.x))
            #     print("Quaternion X : " + str(resp_coordinates.pose.orientation.x))

        except rospy.ServiceException as e:
            rospy.loginfo("Get Model State service call failed:  {0}".format(e))


if __name__ == '__main__':
    tuto = Tutorial()
    tuto.show_gazebo_models()